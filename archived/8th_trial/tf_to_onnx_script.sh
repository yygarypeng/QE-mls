# python -m tf2onnx.convert --saved-model ./ww_resregressor_result/saved_model/ --output ./ww_resregressor_result/ww_resregressor.onnx --opset 17 --verbose
python -m tf2onnx.convert --saved-model ./ww_resregressor_weighted_result/saved_model/ --output ./ww_resregressor_weighted_result/ww_resregressor_weighted.onnx --opset 17 --verbose
