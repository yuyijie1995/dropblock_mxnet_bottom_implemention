
#include "./dropblock-gpu-inl.h"

namespace mxnet {
namespace op {
#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
        using namespace mshadow::cuda;

// The maximum number of blocks to use in the default kernel call.
        constexpr int ROI_MAXIMUM_NUM_BLOCKS = 4096;

/**
 * @brief Compute the number of blocks needed to run N threads.
 */
        inline int ROI_GET_BLOCKS(const int N) {
            return std::max(
                    std::min(
                            (N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock,
                            ROI_MAXIMUM_NUM_BLOCKS),
                    // Use at least 1 block, since CUDA does not allow empty block
                    1);
        }
template <typename T>
__global__ void DropblockForwardKernel(
        const int nthreads,
        const T* input_data,
         T* output_data,
         T* output_mask,
         const int* dev_mask
        ) {
            CUDA_1D_KERNEL_LOOP(index,nthreads){

                //output_mask[index] = dev_mask[index] * (1.0f / pkeep);
                output_mask[index] = dev_mask[index] ;
                output_data[index] = output_mask[index] * input_data[index];


            }

        }

        template<typename T>
        __global__ void DropblockBackwardKernel(
            const int N,//gdata.Size()
            T* gdata,
            const T* grad,
            const T* mask
                ){
            CUDA_1D_KERNEL_LOOP(index,N){
                gdata[index]=grad[index]*mask[index];
            }

        }
     template<typename xpu>
     void DropblockForwardCompute(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<TBlob> &inputs,
             const std::vector<OpReqType>& req,
             const std::vector<TBlob> &outputs){
            using namespace mshadow;
            using namespace mshadow::expr;
            const GPUDropblockParam param =nnvm::get<GPUDropblockParam>(attrs.parsed);
         if (req[gpudropblock::kOut]!=kNullOp)
         {
             CHECK_EQ(inputs.size(),1U);
             if(ctx.is_train)
             {
                 CHECK_EQ(outputs.size(),2U);
             }
             Stream<gpu> *s=ctx.get_stream<gpu>();

             const int count=inputs[gpudropblock::kData].Size();
             const int num_batches=inputs[gpudropblock::kData].shape_[0];
             const int channels=inputs[gpudropblock::kData].shape_[1];
             const int height=inputs[gpudropblock::kData].shape_[2];
             const int width=inputs[gpudropblock::kData].shape_[3];
             const TBlob &out=outputs[gpudropblock::kOut];

             cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);

             if(ctx.is_train||(param.mode==gpudropblock::kAlways))
             {
                 real_t pkeep=param.p;
                 const int blocksize=param.block_size;
                 index_t feat_size=height;
                 double gamma = ((1 - pkeep) / (blocksize * blocksize)) * ((feat_size * feat_size) /((feat_size - blocksize + 1) *
                                                                                                        (feat_size - blocksize + 1)));

                 index_t mask_reduction=blocksize/2;
                 index_t mask_height,mask_width;
                 if ((blocksize % 2) != 0) {
                     mask_height = height - mask_reduction * 2;
                     mask_width = width - mask_reduction * 2;
                 } else {
                     mask_height = height - mask_reduction * 2 + 1;
                     mask_width = width - mask_reduction * 2 + 1;
                 }
                 index_t mask_area = mask_height * mask_width;
                 index_t random_points_num = static_cast<int>(mask_area * gamma);

                 //实现np.arange()操作
                 std::vector<int> a;
                 for (int i = 0; i < mask_area; ++i) {
                     a.push_back(i);
                 }
                 std::vector<std::vector<std::vector<int>>> mask(num_batches, std::vector<std::vector<int >>(1,std::vector<int>(mask_area,
                                                                                                                             0)));
                 //实现random.sample(a,n)的操作
                 for(int i=0;i<random_points_num;)
                 {
                     index_t randnum=rand()%mask_area;
                     if(a[randnum]!=-100)
                     {
                        a[randnum]=-100;
                        ++i;
                     }
                 }
                 for (int i = 0; i < num_batches; ++i) {
                     for (int j = 0; j < mask_area; ++j) {
                         if (a[j] == -100) {
                             mask[i][0][j] = 1;
                         }
                     }
                 }
                 std::vector<std::vector<std::vector<std::vector<int>>>> mask_new(num_batches,
                                                                                  std::vector<std::vector<std::vector<int>>>(
                                                                                          1,
                                                                                          std::vector<std::vector<int>>(
                                                                                                  mask_height,
                                                                                                  std::vector<int>(
                                                                                                          mask_width))));

                 //对应 mask=mask.reshape([data.shape[0], 1, mask_height, mask_width])
                 for (int i = 0; i < num_batches; ++i) {
                     for (int j = 0; j < 1; ++j) {
                         for (int k = 0; k < mask_area; ++k) {
                             index_t mask_i = k / mask_width;
                             index_t mask_j = k % mask_width;
                             mask_new[i][j][mask_i][mask_j] = mask[i][j][k];
                         }
                     }
                 }

                 //生成卷积所使用的weight_mat
                 std::vector<std::vector<std::vector<std::vector<int>>>> weight_mat(num_batches,
                                                                                    std::vector<std::vector<std::vector<int>>>(
                                                                                            1,
                                                                                            std::vector<std::vector<int>>(
                                                                                                    blocksize,
                                                                                                    std::vector<int>(
                                                                                                            blocksize,
                                                                                                            1))));
                 //卷积前的padding操作
                 //根据block_size的不同选择不同的padding策略
                 index_t  padding=0;
                 if(blocksize==3)
                 {
                     padding=blocksize/2 +1;
                 }
                 else if (blocksize==5)
                 {
                     padding=ceil(blocksize/2.0)+1;
                 }
                 else if(blocksize>5)
                 {
                     padding=ceil(blocksize/2.0)+2;
                 }
                 index_t padding_height = mask_height + 2 * padding;
                 index_t padding_width = mask_width + 2 * padding;
                 std::vector<std::vector<std::vector<std::vector<int>>>> mask_padding(num_batches,
                                                                                      std::vector<std::vector<std::vector<int>>>(
                                                                                              1,
                                                                                              std::vector<std::vector<int>>(
                                                                                                      padding_height,
                                                                                                      std::vector<int>(
                                                                                                              padding_width))));
                 for (int i = 0; i < num_batches; ++i) {
                     for (int j = 0; j < 1; ++j) {
                         for (int k = 0; k < padding_height; ++k) {
                             for (int l = 0; l < padding_width; ++l) {
                                 if (k < padding || l < padding) {
                                     mask_padding[i][j][k][l] = 0;
                                 } else if (k > (mask_height + 1) || l > (mask_width + 1)) {
                                     mask_padding[i][j][k][l] = 0;
                                 } else {
                                     mask_padding[i][j][k][l] = mask_new[i][j][k - padding][l - padding];
                                 }
                                 printf("%d\t",mask_padding[i][j][k][l]);
                             }
                             printf("\n");
                         }
                         printf("\n");
                     }
                     printf("\n");
                 }

                 std::vector<std::vector<std::vector<int >>> mask_3d(num_batches, std::vector<std::vector<int >>(1,
                                                                                                              std::vector<int >(
                                                                                                                      padding_height *
                                                                                                                      padding_width)));

                 for (int i = 0; i < num_batches; ++i) {
                     for (int j = 0; j < 1; ++j) {
                         for (int k = 0; k < padding_height; ++k) {
                             for (int l = 0; l < padding_width; ++l) {

                                 mask_3d[i][j][l + k * padding_width] = mask_padding[i][j][k][l];

                             }
                         }
                     }
                 }

                 //把weightmat平铺为三维数组
                 std::vector<std::vector<std::vector<int>>> kernel_3d(num_batches, std::vector<std::vector<int>>(1,
                                                                                                              std::vector<int>(
                                                                                                                      blocksize *
                                                                                                                      blocksize)));

                 for (int i = 0; i < num_batches; ++i) {
                     for (int j = 0; j < 1; ++j) {
                         for (int k = 0; k < blocksize; ++k) {
                             for (int l = 0; l < blocksize; ++l) {

                                 kernel_3d[i][j][l + k * blocksize] = weight_mat[i][j][k][l];
                                 printf("%d\t",kernel_3d[i][j][l + k * blocksize]);
                             }
                             printf("\n");
                         }
                         printf("\n");
                     }
                     printf("\n");
                 }

                 //计算卷积输出矩阵的维数
                 index_t outm = padding_height - blocksize + 1;
                 //计算卷积过程中的被卷积矩阵的宽和高
                 index_t convAw = blocksize * blocksize;
                 index_t convAh = padding_height * padding_width;
                 //定义一个卷积过程中的矩阵
                 std::vector<std::vector<std::vector<int>>> A_convert(num_batches, std::vector<std::vector<int>>(1,
                                                                                                              std::vector<int>(
                                                                                                                      convAh *
                                                                                                                      convAw)));
                 for (int n = 0; n < num_batches; ++n) {
                     for (int j = 0; j < 1; ++j) {
                         for (int k = 0; k < outm; ++k) {
                             for (int m = 0; m < outm; ++m) {
                                 index_t wh = k * outm * convAw + m * convAw;//k*9*9+m*121
                                 index_t col1 = k * padding_height + m;//k*11+m  0
                                 index_t col2 = (k + 1) * padding_height + m;//(k+1)*11+m 11
                                 index_t col3 = (k + 2) * padding_height + m;//(k+2)*11+m  22
                                 index_t col4 = (k + 3) * padding_height + m;//(k+3)*11+m
                                 index_t col5 = (k + 4) * padding_height + m;//(k+4)*11+m
                                 index_t col6 = (k + 5) * padding_height + m;
                                 index_t col7 = (k + 6) * padding_height + m;
                                 if (blocksize == 3) {
                                     A_convert[n][j][wh] = mask_3d[n][j][col1];
                                     A_convert[n][j][wh + 1] = mask_3d[n][j][col1 + 1];
                                     A_convert[n][j][wh + 2] = mask_3d[n][j][col1 + 2];

                                     A_convert[n][j][wh + 3] = mask_3d[n][j][col2];
                                     A_convert[n][j][wh + 4] = mask_3d[n][j][col2 + 1];
                                     A_convert[n][j][wh + 5] = mask_3d[n][j][col2 + 2];

                                     A_convert[n][j][wh + 6] = mask_3d[n][j][col3];
                                     A_convert[n][j][wh + 7] = mask_3d[n][j][col3 + 1];
                                     A_convert[n][j][wh + 8] = mask_3d[n][j][col3 + 2];

                                 } else if (blocksize == 5) {
                                     A_convert[n][j][wh] = mask_3d[n][j][col1];
                                     A_convert[n][j][wh + 1] = mask_3d[n][j][col1 + 1];
                                     A_convert[n][j][wh + 2] = mask_3d[n][j][col1 + 2];
                                     A_convert[n][j][wh + 3] = mask_3d[n][j][col1 + 3];
                                     A_convert[n][j][wh + 4] = mask_3d[n][j][col1 + 4];

                                     A_convert[n][j][wh + 5] = mask_3d[n][j][col2];
                                     A_convert[n][j][wh + 6] = mask_3d[n][j][col2 + 1];
                                     A_convert[n][j][wh + 7] = mask_3d[n][j][col2 + 2];
                                     A_convert[n][j][wh + 8] = mask_3d[n][j][col2 + 3];
                                     A_convert[n][j][wh + 9] = mask_3d[n][j][col2 + 4];

                                     A_convert[n][j][wh + 10] = mask_3d[n][j][col3];
                                     A_convert[n][j][wh + 11] = mask_3d[n][j][col3 + 1];
                                     A_convert[n][j][wh + 12] = mask_3d[n][j][col3 + 2];
                                     A_convert[n][j][wh + 13] = mask_3d[n][j][col3 + 3];
                                     A_convert[n][j][wh + 14] = mask_3d[n][j][col3 + 4];

                                     A_convert[n][j][wh + 15] = mask_3d[n][j][col4];
                                     A_convert[n][j][wh + 16] = mask_3d[n][j][col4 + 1];
                                     A_convert[n][j][wh + 17] = mask_3d[n][j][col4 + 2];
                                     A_convert[n][j][wh + 18] = mask_3d[n][j][col4 + 3];
                                     A_convert[n][j][wh + 19] = mask_3d[n][j][col4 + 4];

                                     A_convert[n][j][wh + 20] = mask_3d[n][j][col5];
                                     A_convert[n][j][wh + 21] = mask_3d[n][j][col5 + 1];
                                     A_convert[n][j][wh + 22] = mask_3d[n][j][col5 + 2];
                                     A_convert[n][j][wh + 23] = mask_3d[n][j][col5 + 3];
                                     A_convert[n][j][wh + 24] = mask_3d[n][j][col5 + 4];
                                 }else if  (blocksize == 7) {
                                     A_convert[n][j][wh] = mask_3d[n][j][col1];
                                     A_convert[n][j][wh + 1] = mask_3d[n][j][col1 + 1];
                                     A_convert[n][j][wh + 2] = mask_3d[n][j][col1 + 2];
                                     A_convert[n][j][wh + 3] = mask_3d[n][j][col1 + 3];
                                     A_convert[n][j][wh + 4] = mask_3d[n][j][col1 + 4];
                                     A_convert[n][j][wh + 5] = mask_3d[n][j][col1 + 5];
                                     A_convert[n][j][wh + 6] = mask_3d[n][j][col1 + 6];

                                     A_convert[n][j][wh + 7] = mask_3d[n][j][col2];
                                     A_convert[n][j][wh + 8] = mask_3d[n][j][col2 + 1];
                                     A_convert[n][j][wh + 9] = mask_3d[n][j][col2 + 2];
                                     A_convert[n][j][wh + 10] = mask_3d[n][j][col2 + 3];
                                     A_convert[n][j][wh + 11] = mask_3d[n][j][col2 + 4];
                                     A_convert[n][j][wh + 12] = mask_3d[n][j][col2 + 5];
                                     A_convert[n][j][wh + 13] = mask_3d[n][j][col2 + 6];

                                     A_convert[n][j][wh + 14] = mask_3d[n][j][col3];
                                     A_convert[n][j][wh + 15] = mask_3d[n][j][col3 + 1];
                                     A_convert[n][j][wh + 16] = mask_3d[n][j][col3 + 2];
                                     A_convert[n][j][wh + 17] = mask_3d[n][j][col3 + 3];
                                     A_convert[n][j][wh + 18] = mask_3d[n][j][col3 + 4];
                                     A_convert[n][j][wh + 19] = mask_3d[n][j][col3 + 5];
                                     A_convert[n][j][wh + 20] = mask_3d[n][j][col3 + 6];

                                     A_convert[n][j][wh + 21] = mask_3d[n][j][col4];
                                     A_convert[n][j][wh + 22] = mask_3d[n][j][col4 + 1];
                                     A_convert[n][j][wh + 23] = mask_3d[n][j][col4 + 2];
                                     A_convert[n][j][wh + 24] = mask_3d[n][j][col4 + 3];
                                     A_convert[n][j][wh + 25] = mask_3d[n][j][col4 + 4];
                                     A_convert[n][j][wh + 26] = mask_3d[n][j][col4 + 5];
                                     A_convert[n][j][wh + 27] = mask_3d[n][j][col4 + 6];

                                     A_convert[n][j][wh + 28] = mask_3d[n][j][col5];
                                     A_convert[n][j][wh + 29] = mask_3d[n][j][col5 + 1];
                                     A_convert[n][j][wh + 30] = mask_3d[n][j][col5 + 2];
                                     A_convert[n][j][wh + 31] = mask_3d[n][j][col5 + 3];
                                     A_convert[n][j][wh + 32] = mask_3d[n][j][col5 + 4];
                                     A_convert[n][j][wh + 33] = mask_3d[n][j][col5 + 5];
                                     A_convert[n][j][wh + 34] = mask_3d[n][j][col5 + 6];

                                     A_convert[n][j][wh + 35] = mask_3d[n][j][col6];
                                     A_convert[n][j][wh + 36] = mask_3d[n][j][col6 + 1];
                                     A_convert[n][j][wh + 37] = mask_3d[n][j][col6 + 2];
                                     A_convert[n][j][wh + 38] = mask_3d[n][j][col6 + 3];
                                     A_convert[n][j][wh + 39] = mask_3d[n][j][col6 + 4];
                                     A_convert[n][j][wh + 40] = mask_3d[n][j][col6 + 5];
                                     A_convert[n][j][wh + 41] = mask_3d[n][j][col6 + 6];

                                     A_convert[n][j][wh + 42] = mask_3d[n][j][col7];
                                     A_convert[n][j][wh + 43] = mask_3d[n][j][col7 + 1];
                                     A_convert[n][j][wh + 44] = mask_3d[n][j][col7 + 2];
                                     A_convert[n][j][wh + 45] = mask_3d[n][j][col7 + 3];
                                     A_convert[n][j][wh + 46] = mask_3d[n][j][col7 + 4];
                                     A_convert[n][j][wh + 47] = mask_3d[n][j][col7 + 5];
                                     A_convert[n][j][wh + 48] = mask_3d[n][j][col7 + 6];
                                 }

                             }
                         }
                     }
                 }
                 std::vector<int> conv_cache;//存储卷积完了的数字
                 for (int i = 0; i < num_batches; ++i) {
                     for (int j = 0; j < 1; ++j) {
                         for (int k = 0; k < outm; ++k) {
                             for (int l = 0; l < outm; ++l) {
                                 int result_one_position = 0;
                                 index_t wh = k * outm * convAw + l * convAw;
                                 for (int m = 0; m < convAw; ++m) {
                                     result_one_position += A_convert[i][j][wh + m] * kernel_3d[i][j][m];
                                 }
                                 conv_cache.push_back(result_one_position);
                             }
                         }
                     }
                 }

                 //把卷积完了的数重组为4维数组
                 std::vector<std::vector<std::vector<std::vector<int>>>> mask_conved(num_batches,
                                                                                     std::vector<std::vector<std::vector<int>>>(
                                                                                             1,
                                                                                             std::vector<std::vector<int>>(
                                                                                                     outm,
                                                                                                     std::vector<int>(
                                                                                                             outm))));

                 index_t delta = blocksize / 2;
                 index_t input_height = mask_height + delta * 2;
                 index_t input_width = mask_width + delta * 2;
                 index_t height_to_crop = outm - input_height;
                 index_t width_to_crop = outm - input_width;

                 if (height_to_crop != 0) {
                     printf("height_to_crop !=0");
                     for (int i = 0; i < num_batches; ++i) {
                         for (int j = 0; j < 1; ++j) {
                             for (int k = 0; k < outm - height_to_crop + 1; ++k) {
                                 printf("\n");
                                 for (int l = 0; l < outm; ++l) {
                                     mask_conved[i][j][k][l] = (conv_cache[i * outm * (outm - height_to_crop)
                                                                  + j * outm * (outm - height_to_crop) +
                                                                  k * outm + l]==0)? 1:0;
                                     printf("%d\t",conv_cache[i * outm * (outm - height_to_crop)
                                                     + j * outm * (outm - height_to_crop) +
                                                     k * outm + l]);
                                 }

                             }
                         }
                     }
                 }
                 if (width_to_crop != 0) {
                     printf("width_to_crop !=0");
                     for (int i = 0; i < num_batches; ++i) {
                         for (int j = 0; j < 1; ++j) {
                             for (int k = 0; k < outm; ++k) {
                                 printf("\n");
                                 for (int l = 0; l < outm - width_to_crop + 1; ++l) {
                                     mask_conved[i][j][k][l] =( conv_cache[i * outm * (outm - width_to_crop) +
                                                                  j * outm * (outm - width_to_crop) +
                                                                  k * (outm - width_to_crop) + l]==0)? 1:0;
                                     printf("%d\t",conv_cache[i * outm * (outm - width_to_crop) +
                                                     j * outm * (outm - width_to_crop) +
                                                     k * (outm - width_to_crop) + l]);
                                 }
                             }
                         }
                     }
                 }
                 if ((width_to_crop != 0)&&(height_to_crop!=0)) {
                     printf("width_to_crop !=0");
                     for (int i = 0; i < num_batches; ++i) {
                         for (int j = 0; j < 1; ++j) {
                             for (int k = 0; k < outm-height_to_crop+1; ++k) {
                                 printf("\n");
                                 for (int l = 0; l < outm - width_to_crop + 1; ++l) {
                                     mask_conved[i][j][k][l] =( conv_cache[i * (outm-height_to_crop) * (outm - width_to_crop) +
                                                                  j * (outm-height_to_crop) * (outm - width_to_crop) +
                                                                  k * (outm - width_to_crop) + l]==0)? 1:0;
                                     printf("%d\t",conv_cache[i * outm * (outm - width_to_crop) +
                                                     j * outm * (outm - width_to_crop) +
                                                     k * (outm - width_to_crop) + l]);
                                 }
                             }
                         }
                     }
                 }

                 printf("width_to_crop和height_to_crop都等于0");
                 for (int i = 0; i < num_batches; ++i) {
                     for (int j = 0; j < 1; ++j) {
                         for (int k = 0; k < outm; ++k) {
                             printf("\n");
                             for (int l = 0; l < outm; ++l) {
                                 mask_conved[i][j][k][l] =(conv_cache[i * outm * outm + j * outm * outm + k * outm + l]==0)? 1:0;

                                 printf("%d\t",conv_cache[i * outm * outm + j * outm * outm + k * outm + l]);
                             }
                             printf("\n");
                         }
                         printf("\n");
                     }
                     printf("\n");
                 }
                 //把mask_conved变为一个1D的数组来与indata进行计算
                 //std::vector<int> mask_conved_1d(num_batches * channels * height * width);
                 int mask_conved_1d[count];
                 int *dev_mask;
                 printf("mask_conved_1d：\n");
                 for (int i = 0; i < num_batches; ++i) {
                     for (int j = 0; j < channels; ++j) {
                         for (int k = 0; k < height; ++k) {
                             for (int l = 0; l < width; ++l) {
                                 mask_conved_1d[i * channels * height * width
                                                + j * height * width +
                                                k * width + l] = mask_conved[i][0][k][l];
                                 printf("%d\t",mask_conved_1d[i * channels * height * width
                                                              + j * height * width +
                                                              k * width + l]);
                             }
                             printf("\n");
                         }
                         printf("\n");
                     }
                 }
                 //allocate memory on GPU
                 cudaMalloc((void**)&dev_mask,count* sizeof(int));

                 cudaMemcpy(dev_mask,mask_conved_1d,count* sizeof(int),cudaMemcpyHostToDevice);

                 MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_,DType,{
                     const DType *input_data=inputs[gpudropblock::kData].dptr<DType>();
                     DType *mask_out=outputs[gpudropblock::kMask].dptr<DType>();
                     DType *dropblock_out=outputs[gpudropblock::kOut].dptr<DType>();
//                     DropblockForwardKernel<DType><<<ROI_GET_BLOCKS(count),kMaxThreadsPerBlock,0,stream>>>(
//                             count, param.block_size,num_batches,channels,height,width,param.p,
//                                     input_data,dropblock_out,mask_out
//                     );
                     DropblockForwardKernel<DType><<<ROI_GET_BLOCKS(count),kMaxThreadsPerBlock,0,stream>>>(
                             count,input_data,dropblock_out,mask_out,dev_mask
                     );
                 })
             }
             else{
                 //printf("以上啥都没有发生");
                 const TBlob& data = inputs[gpudropblock::kData];
                 if (req[gpudropblock::kOut] == kWriteTo) {
                     mxnet_op::copy(s, out, data);
                 } else {
                     MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_,DType,{MXNET_ASSIGN_REQ_SWITCH(req[gpudropblock::kOut], Req, {
                                 mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
                                         s, out.Size(), out.dptr<DType>(), data.dptr<DType>());//identity:input==output
                         });
                     })

                 }
             }

         }

        }

        template <typename xpu>
        void DropblockBackwardCompute(const nnvm::NodeAttrs& attrs,
                const OpContext& ctx,
                const std::vector<TBlob> &inputs,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &outputs){
            CHECK_EQ(inputs.size(),2U);
            CHECK_EQ(outputs.size(),1);
            CHECK_EQ(req.size(),1);
            using namespace mshadow;
            using namespace mshadow::expr;
            std::vector<TBlob> out_grads(2);
            std::vector<TBlob> out_data(2);
            out_grads[gpudropblock::kOut]=inputs[0];
            out_data[gpudropblock::kMask]=inputs[1];
            Stream<gpu> *s=ctx.get_stream<gpu>();
            cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
            const GPUDropblockParam param =nnvm::get<GPUDropblockParam>(attrs.parsed);
            if(ctx.is_train||param.mode==gpudropblock::kAlways)
            {
                const TBlob &gdata=outputs[gpudropblock::kData];
                const TBlob &grad=out_grads[gpudropblock::kOut];
                const TBlob &mask=out_data[gpudropblock::kMask];
                const int count=inputs[gpudropblock::kData].Size();

                MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_,DType,{
                    DropblockBackwardKernel<DType>
                            <<<ROI_GET_BLOCKS(count),
                    kMaxThreadsPerBlock,
                    0,
                    stream>>>(
                            count, gdata.dptr<DType>(),
                                    grad.dptr<DType>(), mask.dptr<DType>()
                    );
                })
            }else{
                const TBlob& gdata = outputs[gpudropblock::kData];
                const TBlob& grad = out_grads[gpudropblock::kOut];
                if (req[gpudropblock::kData] == kWriteTo) {
                    mxnet_op::copy(s, gdata, grad);
                } else {
                    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_,DType,{MXNET_ASSIGN_REQ_SWITCH(req[gpudropblock::kData], Req, {
                                mxnet_op::Kernel<mxnet_op::op_with_req<mshadow_op::identity, Req>, xpu>::Launch(
                                        s, gdata.Size(), gdata.dptr<DType>(), grad.dptr<DType>());
                        });})

                }
            }




        }



NNVM_REGISTER_OP(GPUDropblock)
.set_attr<FCompute>("FCompute<gpu>", DropblockForwardCompute<gpu>);

NNVM_REGISTER_OP(_backward_GPUDropout)
.set_attr<FCompute>("FCompute<gpu>", DropblockBackwardCompute<gpu>);

}  // namespace op
}  // namespace mxnet


