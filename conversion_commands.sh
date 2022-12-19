
MODEL_NAME=HITNet_SF_oak_sized_model

# ONNX conversion based on synthetic images
deactivate
cd /home/bz/Documents/SpatialAI/TinyHITNet
python3 onnx_convert_synth.py --model HITNet_SF --ckpt /home/bz/Documents/SpatialAI/TinyHITNet/ckpt/old/hitnet_sf_finalpass.ckpt --model_name $MODEL_NAME

# MODEL OPTIMIZATION => IR
source ~/visitor_patch/bin/activate
cd ~/hitnet_out/onnxs/

mo --input_model ${MODEL_NAME}.onnx --data_type=FP16 --mean_values=left[127.5,127.5,127.5],right[127.5,127.5,127.5] --scale_values=left[255,255,255],right[255,255,255]


OPENVINO_PATH=~/visitor_patch/openvino_range_patch
source $OPENVINO_PATH/setupvars.sh
COMPILE_TOOL="$OPENVINO_PATH/tools/compile_tool/compile_tool"

# MYRIAD COMPILE
$COMPILE_TOOL -m $MODEL_NAME.xml -ip U8 -d MYRIAD -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6 -c myriad.conf

# CPU Compile
$COMPILE_TOOL -m $MODEL_NAME.xml -ip U8 -d CPU -o ${MODEL_NAME}_cpu.blob





