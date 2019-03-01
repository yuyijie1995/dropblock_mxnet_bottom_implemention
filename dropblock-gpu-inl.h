//
// Created by yijie.yu on 2019/2/25.
//

#ifndef DROPBLOCK_CPP_DROPBLOCK_GPU_H
#define DROPBLOCK_CPP_DROPBLOCK_GPU_H


#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../tensor/init_op.h"
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <string>
#include <algorithm>
#include "../mxnet_op.h"
#include "../random/sampler.h"
#include "../tensor/elemwise_binary_broadcast_op.h"


namespace mxnet {
    namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
        namespace gpudropblock {
            enum GPUDropblockOpInputs {kData};
            enum GPUDropblockOpOutputs {kOut,kMask};
            enum GPUDropblockOpForwardResource {kRandom};
            enum GPUDropblockOpMode {kTraining,kAlways};
        }  // namespace dropblock


        struct GPUDropblockParam : public dmlc::Parameter<GPUDropblockParam> {
            real_t p;
            int mode;
            int block_size;
            TShape axes;
            DMLC_DECLARE_PARAMETER(GPUDropblockParam) {
                DMLC_DECLARE_FIELD(p).set_default(0.5)
                        .set_range(0,1)
                        .describe("Fraction of the input that gets dropped out during training time.");
                DMLC_DECLARE_FIELD(block_size).set_default(3)
                        .describe("the block size");
                DMLC_DECLARE_FIELD(mode)
                        .add_enum("training",gpudropblock::kTraining)
                        .add_enum("always",gpudropblock::kAlways)
                        .set_default(gpudropblock::kTraining)
                        .describe("Whether to only turn on dropblock during training or to also turn on for inference.");
                DMLC_DECLARE_FIELD(axes).set_default(TShape())
                        .describe("Axes for variational dropblock kernel.");
            }
        };
//        static OpStatePtr CreateDropoutState(const nnvm::NodeAttrs &attrs,
//                                             const Context ctx,
//                                             const std::vector<TShape> &in_shapes,
//                                             const std::vector<int> &in_types) {
//            const GPUDropblockParam& param = nnvm::get<GPUDropblockParam>(attrs.parsed);
//            OpStatePtr state;
//            MSHADOW_REAL_TYPE_SWITCH(in_types[gpudropblock::kData], DType, {
//                if (ctx.dev_type == kGPU) {
//                    state = OpStatePtr::Create<GPUDropblockOp<gpu, DType>>(param, ctx);
//                } else {
//                    state = OpStatePtr::Create<GPUdropblockOp<cpu, DType>>(param, ctx);
//                }
//                return state;
//            });
//            LOG(FATAL) << "should never reach here";
//            return OpStatePtr();  // should never reach here
//        }

    }  // namespace op
}  // namespace mxnet










#endif //DROPBLOCK_CPP_DROPBLOCK_GPU_H
