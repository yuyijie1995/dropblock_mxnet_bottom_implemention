
#include "./dropblock-gpu-inl.h"
#include "../operator_common.h"

namespace mxnet {
namespace op {


template<typename T>
void DropblockForward(
        const int N,
        const int nbatches,
        const int nchannel,
        const int nheight,
        const int nwidth,
        const T* input_data,
        T* mask_out,
        T* dropblock_out,
        real_t  pkeep,
        const int block_size
        )
{
    DCHECK(nbatches!=0);
    DCHECK(N>0);
    index_t feat_size=nheight;
    double gamma=((1 - pkeep) / (block_size * block_size)) * ((feat_size * feat_size) /
                                                              ((feat_size - block_size + 1) *
                                                               (feat_size - block_size + 1)));
    index_t mask_reduction = block_size / 2;
    index_t mask_height, mask_width;
    if ((block_size % 2) != 0) {
        mask_height = nheight - mask_reduction * 2;
        mask_width = nwidth - mask_reduction * 2;
    } else {
        mask_height = nheight - mask_reduction * 2 + 1;
        mask_width = nwidth - mask_reduction * 2 + 1;
    }
    index_t mask_area = mask_height * mask_width;
    index_t n = static_cast<int>(mask_area * gamma);

    //实现np.arange()操作
    std::vector<int> a;
    for (int j = 0; j < mask_area; ++j) {
        a.push_back(j);
    }

    //声明动态三维数组
    std::vector<std::vector<std::vector<int>>> mask(nbatches, std::vector<std::vector<int >>(1,std::vector<int>(mask_area,
                                                                                                                0)));
    //实现random.sample(a,n)的操作
    int l = 0;
    while (l < n) {
        index_t randnum = rand() % mask_area;

        if (a[randnum] != -100) {
            a[randnum] = -100;
            ++l;
        }
    }

    printf("mask:\n");
    for (int j = 0; j < nbatches; ++j) {
        for (int k = 0; k < mask_area; ++k) {
            if (a[k] == -100) {
                mask[j][0][k] = 1;
            }
            printf("%d\t",mask[j][0][k]);
        }
        printf("\n");
    }
    std::vector<std::vector<std::vector<std::vector<int>>>> mask_new(nbatches,
                                                                     std::vector<std::vector<std::vector<int>>>(
                                                                             1,
                                                                             std::vector<std::vector<int>>(
                                                                                     mask_height,
                                                                                     std::vector<int>(
                                                                                             mask_width))));

    index_t mask_i = 0;
    index_t mask_j = 0;
    printf("mask_new:\n");
    for (int m = 0; m < nbatches; ++m) {
        for (int n = 0; n < 1; ++n) {
            for (int l = 0; l < mask_area; ++l) {
                mask_i = l / mask_width;
                mask_j = l % mask_width;
                mask_new[m][n][mask_i][mask_j] = mask[m][n][l];
                printf("%d\t",mask_new[m][n][mask_i][mask_j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    //生成卷积所使用的weight_mat
    std::vector<std::vector<std::vector<std::vector<int>>>> weight_mat(nbatches,
                                                                       std::vector<std::vector<std::vector<int>>>(
                                                                               1,
                                                                               std::vector<std::vector<int>>(
                                                                                       block_size,
                                                                                       std::vector<int>(
                                                                                               block_size,
                                                                                               1))));
    index_t  padding=0;
    if(block_size==3)
    {
        padding=block_size/2 +1;
    }
    else if (block_size==5)
    {
        padding=ceil(block_size/2.0)+1;
    }
    else if(block_size>5)
    {
        padding=ceil(block_size/2.0)+2;
    }
    index_t padding_height = mask_height + 2 * padding;
    index_t padding_width = mask_width + 2 * padding;
    std::vector<std::vector<std::vector<std::vector<int>>>> mask_padding(nbatches,
                                                                         std::vector<std::vector<std::vector<int>>>(
                                                                                 1,
                                                                                 std::vector<std::vector<int>>(
                                                                                         padding_height,
                                                                                         std::vector<int>(
                                                                                                 padding_width))));
    printf("mask_padding:\n");
    for (int n = 0; n < nbatches; ++n) {
        for (int j = 0; j < 1; ++j) {
            for (int k = 0; k < padding_height; ++k) {
                for (int m = 0; m < padding_width; ++m) {
                    if (k < padding || m < padding) {
                        mask_padding[n][j][k][m] = 0;
                    } else if (k > (mask_height + 1) || m > (mask_width + 1)) {
                        mask_padding[n][j][k][m] = 0;
                    } else {
                        mask_padding[n][j][k][m] = mask_new[n][j][k - padding][m - padding];
                    }
                    printf("%d\t",mask_padding[n][j][k][m]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
    std::vector<std::vector<std::vector<int >>> mask_1d(nbatches,std::vector<std::vector<int>>(1,std::vector<int>(padding_height*padding_width)));

    for (int n = 0; n < nbatches; ++n) {
        for (int j = 0; j < 1; ++j) {
            for (int k = 0; k < padding_height; ++k) {
                for (int m = 0; m < padding_width; ++m) {

                    mask_1d[n][j][m + k * padding_width] = mask_padding[n][j][k][m];

                }
            }
        }
    }
    std::vector<std::vector<std::vector<int>>> kernel_1d(nbatches,std::vector<std::vector<int>>(1,std::vector<int>(block_size*block_size)));
    printf("kernel_1d:\n");
    for (int n = 0; n < nbatches; ++n) {
        for (int j = 0; j < 1; ++j) {
            for (int k = 0; k < block_size; ++k) {
                for (int m = 0; m < block_size; ++m) {

                    kernel_1d[n][j][m + k * block_size] = weight_mat[n][j][k][m];
                    printf("%d\t",kernel_1d[n][j][m + k * block_size]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }

    //计算卷积输出矩阵的维数
    index_t outm = padding_height - block_size + 1;
    //计算卷积过程中的被卷积矩阵的宽和高
    index_t convAw = block_size * block_size;
    index_t convAh = padding_height * padding_width;
    //定义一个卷积过程中的矩阵
    std::vector<std::vector<std::vector<int>>> A_convert(nbatches, std::vector<std::vector<int>>(1,
                                                                                                 std::vector<int>(
                                                                                                         convAh *
                                                                                                         convAw)));
    for (int n = 0; n < nbatches; ++n) {
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
                    if (block_size == 3) {
                        A_convert[n][j][wh] = mask_1d[n][j][col1];
                        A_convert[n][j][wh + 1] = mask_1d[n][j][col1 + 1];
                        A_convert[n][j][wh + 2] = mask_1d[n][j][col1 + 2];

                        A_convert[n][j][wh + 3] = mask_1d[n][j][col2];
                        A_convert[n][j][wh + 4] = mask_1d[n][j][col2 + 1];
                        A_convert[n][j][wh + 5] = mask_1d[n][j][col2 + 2];

                        A_convert[n][j][wh + 6] = mask_1d[n][j][col3];
                        A_convert[n][j][wh + 7] = mask_1d[n][j][col3 + 1];
                        A_convert[n][j][wh + 8] = mask_1d[n][j][col3 + 2];

                    } else if (block_size == 5) {
                        A_convert[n][j][wh] = mask_1d[n][j][col1];
                        A_convert[n][j][wh + 1] = mask_1d[n][j][col1 + 1];
                        A_convert[n][j][wh + 2] = mask_1d[n][j][col1 + 2];
                        A_convert[n][j][wh + 3] = mask_1d[n][j][col1 + 3];
                        A_convert[n][j][wh + 4] = mask_1d[n][j][col1 + 4];

                        A_convert[n][j][wh + 5] = mask_1d[n][j][col2];
                        A_convert[n][j][wh + 6] = mask_1d[n][j][col2 + 1];
                        A_convert[n][j][wh + 7] = mask_1d[n][j][col2 + 2];
                        A_convert[n][j][wh + 8] = mask_1d[n][j][col2 + 3];
                        A_convert[n][j][wh + 9] = mask_1d[n][j][col2 + 4];

                        A_convert[n][j][wh + 10] = mask_1d[n][j][col3];
                        A_convert[n][j][wh + 11] = mask_1d[n][j][col3 + 1];
                        A_convert[n][j][wh + 12] = mask_1d[n][j][col3 + 2];
                        A_convert[n][j][wh + 13] = mask_1d[n][j][col3 + 3];
                        A_convert[n][j][wh + 14] = mask_1d[n][j][col3 + 4];

                        A_convert[n][j][wh + 15] = mask_1d[n][j][col4];
                        A_convert[n][j][wh + 16] = mask_1d[n][j][col4 + 1];
                        A_convert[n][j][wh + 17] = mask_1d[n][j][col4 + 2];
                        A_convert[n][j][wh + 18] = mask_1d[n][j][col4 + 3];
                        A_convert[n][j][wh + 19] = mask_1d[n][j][col4 + 4];

                        A_convert[n][j][wh + 20] = mask_1d[n][j][col5];
                        A_convert[n][j][wh + 21] = mask_1d[n][j][col5 + 1];
                        A_convert[n][j][wh + 22] = mask_1d[n][j][col5 + 2];
                        A_convert[n][j][wh + 23] = mask_1d[n][j][col5 + 3];
                        A_convert[n][j][wh + 24] = mask_1d[n][j][col5 + 4];
                    }else if  (block_size == 7) {
                        A_convert[n][j][wh] = mask_1d[n][j][col1];
                        A_convert[n][j][wh + 1] = mask_1d[n][j][col1 + 1];
                        A_convert[n][j][wh + 2] = mask_1d[n][j][col1 + 2];
                        A_convert[n][j][wh + 3] = mask_1d[n][j][col1 + 3];
                        A_convert[n][j][wh + 4] = mask_1d[n][j][col1 + 4];
                        A_convert[n][j][wh + 5] = mask_1d[n][j][col1 + 5];
                        A_convert[n][j][wh + 6] = mask_1d[n][j][col1 + 6];

                        A_convert[n][j][wh + 7] = mask_1d[n][j][col2];
                        A_convert[n][j][wh + 8] = mask_1d[n][j][col2 + 1];
                        A_convert[n][j][wh + 9] = mask_1d[n][j][col2 + 2];
                        A_convert[n][j][wh + 10] = mask_1d[n][j][col2 + 3];
                        A_convert[n][j][wh + 11] = mask_1d[n][j][col2 + 4];
                        A_convert[n][j][wh + 12] = mask_1d[n][j][col2 + 5];
                        A_convert[n][j][wh + 13] = mask_1d[n][j][col2 + 6];

                        A_convert[n][j][wh + 14] = mask_1d[n][j][col3];
                        A_convert[n][j][wh + 15] = mask_1d[n][j][col3 + 1];
                        A_convert[n][j][wh + 16] = mask_1d[n][j][col3 + 2];
                        A_convert[n][j][wh + 17] = mask_1d[n][j][col3 + 3];
                        A_convert[n][j][wh + 18] = mask_1d[n][j][col3 + 4];
                        A_convert[n][j][wh + 19] = mask_1d[n][j][col3 + 5];
                        A_convert[n][j][wh + 20] = mask_1d[n][j][col3 + 6];

                        A_convert[n][j][wh + 21] = mask_1d[n][j][col4];
                        A_convert[n][j][wh + 22] = mask_1d[n][j][col4 + 1];
                        A_convert[n][j][wh + 23] = mask_1d[n][j][col4 + 2];
                        A_convert[n][j][wh + 24] = mask_1d[n][j][col4 + 3];
                        A_convert[n][j][wh + 25] = mask_1d[n][j][col4 + 4];
                        A_convert[n][j][wh + 26] = mask_1d[n][j][col4 + 5];
                        A_convert[n][j][wh + 27] = mask_1d[n][j][col4 + 6];

                        A_convert[n][j][wh + 28] = mask_1d[n][j][col5];
                        A_convert[n][j][wh + 29] = mask_1d[n][j][col5 + 1];
                        A_convert[n][j][wh + 30] = mask_1d[n][j][col5 + 2];
                        A_convert[n][j][wh + 31] = mask_1d[n][j][col5 + 3];
                        A_convert[n][j][wh + 32] = mask_1d[n][j][col5 + 4];
                        A_convert[n][j][wh + 33] = mask_1d[n][j][col5 + 5];
                        A_convert[n][j][wh + 34] = mask_1d[n][j][col5 + 6];

                        A_convert[n][j][wh + 35] = mask_1d[n][j][col6];
                        A_convert[n][j][wh + 36] = mask_1d[n][j][col6 + 1];
                        A_convert[n][j][wh + 37] = mask_1d[n][j][col6 + 2];
                        A_convert[n][j][wh + 38] = mask_1d[n][j][col6 + 3];
                        A_convert[n][j][wh + 39] = mask_1d[n][j][col6 + 4];
                        A_convert[n][j][wh + 40] = mask_1d[n][j][col6 + 5];
                        A_convert[n][j][wh + 41] = mask_1d[n][j][col6 + 6];

                        A_convert[n][j][wh + 42] = mask_1d[n][j][col7];
                        A_convert[n][j][wh + 43] = mask_1d[n][j][col7 + 1];
                        A_convert[n][j][wh + 44] = mask_1d[n][j][col7 + 2];
                        A_convert[n][j][wh + 45] = mask_1d[n][j][col7 + 3];
                        A_convert[n][j][wh + 46] = mask_1d[n][j][col7 + 4];
                        A_convert[n][j][wh + 47] = mask_1d[n][j][col7 + 5];
                        A_convert[n][j][wh + 48] = mask_1d[n][j][col7 + 6];
                    }

                }
            }
        }
    }

    std::vector<int> C;//存储卷积完了的数字
    //printf("详细的计算信息：");
    for (int k = 0; k < nbatches; ++k) {
        for (int l = 0; l < 1; ++l) {
            for (int m = 0; m < outm; ++m) {
                for (int n = 0; n < outm; ++n) {
                    int result_one_position = 0;
                    index_t wh = m * outm * convAw + n * convAw;
                    for (int j = 0; j < convAw; ++j) {
                        result_one_position += A_convert[k][l][wh + j] * kernel_1d[k][l][j];
                        //printf("A_convert[%d][%d][%lld]:%d\t",k,l,wh+j,A_convert[k][l][wh + j]);
                        //printf("kernel_1d[%d][%d][%d]:%d\t",k,l,j,kernel_1d[k][l][j]);

                    }
                    //printf("\n");
                    C.push_back(result_one_position);//input[0]*1*outm*outm
                }
            }
        }
    }

    //重组为4维数组
    std::vector<std::vector<std::vector<std::vector<int>>>> mask_conved(nbatches,
                                                                        std::vector<std::vector<std::vector<int>>>(
                                                                                1,
                                                                                std::vector<std::vector<int>>(
                                                                                        outm,
                                                                                        std::vector<int>(
                                                                                                outm))));

    index_t delta = block_size / 2;
    index_t input_height = mask_height + delta * 2;
    index_t input_width = mask_width + delta * 2;
    index_t height_to_crop = outm - input_height;
    index_t width_to_crop = outm - input_width;
    if (height_to_crop != 0) {
        printf("height_to_crop !=0");
        for (int k = 0; k < nbatches; ++k) {
            for (int l = 0; l < 1; ++l) {
                for (int m = 0; m < outm - height_to_crop + 1; ++m) {
                    printf("\n");
                    for (int n = 0; n < outm; ++n) {
                        mask_conved[k][l][m][n] = (C[k * outm * (outm - height_to_crop)
                                                     + l * outm * (outm - height_to_crop) +
                                                     m * outm + n]==0)? 1:0;
                        printf("%d\t",C[k * outm * (outm - height_to_crop)
                                        + l * outm * (outm - height_to_crop) +
                                        m * outm + n]);
                    }

                }
            }
        }
    }
    if (width_to_crop != 0) {
        printf("width_to_crop !=0");
        for (int k = 0; k < nbatches; ++k) {
            for (int l = 0; l < 1; ++l) {
                for (int m = 0; m < outm; ++m) {
                    printf("\n");
                    for (int n = 0; n < outm - width_to_crop + 1; ++n) {
                        mask_conved[k][l][m][n] =( C[k * outm * (outm - width_to_crop) +
                                                     l * outm * (outm - width_to_crop) +
                                                     m * (outm - width_to_crop) + n]==0)? 1:0;
                        printf("%d\t",C[k * outm * (outm - width_to_crop) +
                                        l * outm * (outm - width_to_crop) +
                                        m * (outm - width_to_crop) + n]);
                    }
                }
            }
        }
    }
    if ((width_to_crop != 0)&&(height_to_crop!=0)) {
        printf("width_to_crop !=0");
        for (int k = 0; k < nbatches; ++k) {
            for (int l = 0; l < 1; ++l) {
                for (int m = 0; m < outm-height_to_crop+1; ++m) {
                    printf("\n");
                    for (int n = 0; n < outm - width_to_crop + 1; ++n) {
                        mask_conved[k][l][m][n] =( C[k * (outm-height_to_crop) * (outm - width_to_crop) +
                                                     l * (outm-height_to_crop) * (outm - width_to_crop) +
                                                     m * (outm - width_to_crop) + n]==0)? 1:0;
                        printf("%d\t",C[k * outm * (outm - width_to_crop) +
                                        l * outm * (outm - width_to_crop) +
                                        m * (outm - width_to_crop) + n]);
                    }
                }
            }
        }
    }
    printf("width_to_crop和height_to_crop都等于0");
    for (int k = 0; k < nbatches; ++k) {
        for (int l = 0; l < 1; ++l) {
            for (int m = 0; m < outm; ++m) {
                printf("\n");
                for (int n = 0; n < outm; ++n) {
                    mask_conved[k][l][m][n] =(C[k * outm * outm + l * outm * outm + m * outm + n]==0)? 1:0;

                    printf("%d\t",C[k * outm * outm + l * outm * outm + m * outm + n]);
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\n");
    //把mask_conved变为一个1D的数组来与indata进行计算
    std::vector<int> mask_conved_1d(nbatches * nchannel * nheight * nwidth);
    printf("mask_conved_1d：\n");
    for (int k = 0; k < nbatches; ++k) {
        for (int l = 0; l < nchannel; ++l) {
            for (int m = 0; m < nheight; ++m) {

                for (int n = 0; n < nwidth; ++n) {
                    mask_conved_1d[k * nchannel * nheight * nwidth
                                   + l * nheight * nwidth +
                                   m * nwidth + n] = mask_conved[k][0][m][n];
                    printf("%d\t",mask_conved_1d[k * nchannel * nheight * nwidth
                                                 + l * nheight * nwidth +
                                                 m * nwidth + n]);

                }
                printf("\n");
            }
            printf("\n");
        }
    }
    for (index_t i = 0;  i < N; ++i) {
        mask_out[i] = mask_conved_1d[i] * (1.0f / pkeep);
        dropblock_out[i] = mask_out[i] * input_data[i];
    }


}

template <typename T>
void DropblockBackward(const index_t N, T* pgdata ,const T* pgrad, const T* pmask){
    for (size_t i=0;i<N;++i)
    {
        pgdata[i]=pgrad[i]*pmask[i];
    }
}





template<typename xpu>
void DropblockForwardCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs)
{
    using namespace mshadow;
    const GPUDropblockParam& param=nnvm::get<GPUDropblockParam>(attrs.parsed);
    if (req[gpudropblock::kOut]!=kNullOp)
    {
        CHECK_EQ(inputs.size(),1U);
        if(ctx.is_train) {
            CHECK_EQ(outputs.size(),2U);
        }
        Stream<cpu> *s=ctx.get_stream<cpu>();

        const int count=outputs[gpudropblock::kData].Size();
     const int nbatches=inputs[gpudropblock::kData].shape_[0];
     const int nchannel=inputs[gpudropblock::kData].shape_[1];
     const int nheight=inputs[gpudropblock::kData].shape_[2];
     const int nwidth=inputs[gpudropblock::kData].shape_[3];
     const TBlob &out=outputs[gpudropblock::kOut];
     if(ctx.is_train||(param.mode==gpudropblock::kAlways))
     {
         MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_,DType,{
             const DType *input_data=inputs[gpudropblock::kData].dptr<DType>();
             DType *mask_out=outputs[gpudropblock::kMask].dptr<DType>();
             DType *dropblock_out=outputs[gpudropblock::kOut].dptr<DType>();
             DropblockForward<DType>(count,nbatches,nchannel,nheight,nwidth,
                                     input_data,mask_out,dropblock_out,param.p,param.block_size);
         })
     }
     else {
         printf("以上啥都没有发生");
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



    template<typename xpu>
    void DropblockBackwardCompute(const nnvm::NodeAttrs& attrs,
            const OpContext& ctx,
            const std::vector<TBlob> &inputs,
            const std::vector<OpReqType> &req,
            const std::vector<TBlob> &outputs
            )
            {
        const GPUDropblockParam& param =nnvm::get<GPUDropblockParam>(attrs.parsed);
        CHECK_EQ(inputs.size(),2U);
        CHECK_EQ(outputs.size(),1);
        CHECK_EQ(req.size(),1);
        std::vector<TBlob> out_grads(2);
        std::vector<TBlob> out_data(2);
        out_grads[gpudropblock::kOut]=inputs[0];
        out_data[gpudropblock::kMask]=inputs[1];
        using namespace mshadow;
        using namespace mshadow::expr;
        Stream<cpu> *s =ctx.get_stream<cpu>();
        if(ctx.is_train || param.mode==gpudropblock::kAlways){
            const TBlob &gdata=outputs[gpudropblock::kData];
            const TBlob &grad=out_grads[gpudropblock::kOut];
            const TBlob &mask=out_data[gpudropblock::kMask];
            const index_t count =gdata.Size();
            MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_,DType,{
                 DType *pgdata=gdata.dptr<DType>();
                 const DType *pgrad=grad.dptr<DType>();
                 const DType *pmask=mask.dptr<DType>();
                 DropblockBackward <DType>(count,pgdata,pgrad,pmask);

            })
        } else{
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

DMLC_REGISTER_PARAMETER(GPUDropblockParam);

NNVM_REGISTER_OP(GPUDropblock)//这个就是OP的名字么？
.describe(R"(Applies dropblock operation to input array.

- During training, each element of the input is set to zero with probability p.
  The whole array is rescaled by :math:`1/(1-p)` to keep the expected
  sum of the input unchanged.

- During testing, this operator does not change the input if mode is 'training'.
  If mode is 'always', the same computaion as during training will be applied.

Example::

  random.seed(998)
  input_array = array([[3., 0.5,  -0.5,  2., 7.],
                      [2., -0.4,   7.,  3., 0.2]])
  a = symbol.Variable('a')
  dropblock = symbol.GPUDropblock(a, p = 0.2,block_size=3)
  executor = dropblock.simple_bind(a = input_array.shape)

  ## If training
  executor.forward(is_train = True, a = input_array)
  executor.outputs
  [[ 3.75   0.625 -0.     2.5    8.75 ]
   [ 2.5   -0.5    8.75   3.75   0.   ]]

  ## If testing
  executor.forward(is_train = False, a = input_array)
  executor.outputs
  [[ 3.     0.5   -0.5    2.     7.   ]
   [ 2.    -0.4    7.     3.     0.2  ]]
)" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr_parser(ParamParser<GPUDropblockParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "mask"};
})
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      std::vector<TShape> *in_shape, std::vector<TShape> *out_shape){
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 1U);
  const GPUDropblockParam& param = nnvm::get<GPUDropblockParam>(attrs.parsed);
  TShape dshape(in_shape->at(0));
  if (dshape.ndim() == 0) return false;
  TShape datashape=in_shape->at(gpudropblock::kData);
  CHECK_EQ(datashape.ndim(),4)<<"data should be a 4D tensor";
  out_shape->clear();
  out_shape->push_back(dshape);
  for (index_t i = 0; i < param.axes.ndim(); ++i) {
    dshape[param.axes[i]] = 1;
  }
  out_shape->push_back(dshape);
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 1U);
  int dtype = in_type->at(0);

  if (dtype == -1) {
    LOG(FATAL) << "input type to dropblock is not specified.";
    return false;
  }

  size_t nout = 2;
  out_type->clear();
  for (size_t i = 0; i < nout; ++i) out_type->push_back(dtype);
  return true;
})
//.set_attr<FCreateOpState>("FCreateOpState", CreateDropoutState)
.set_attr<FCompute>("FCompute<cpu>",DropblockForwardCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
                           [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {

    std::vector<nnvm::NodeEntry> heads;
                               heads.push_back(ograds[0]);
                               heads.emplace_back(nnvm::NodeEntry{n, gpudropblock::kMask, 0});
                               return MakeGradNode("_backward_GPUDropblock", n, heads, n->attrs.dict);
                           })
.add_argument("data", "NDArray-or-Symbol", "Input array to which dropout will be applied.")
.add_arguments(GPUDropblockParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_GPUDropblock)
.set_attr_parser(ParamParser<GPUDropblockParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", DropblockBackwardCompute<cpu>);
}  // namespace op
}  // namespace mxnet
