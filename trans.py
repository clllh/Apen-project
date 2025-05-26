import onnx
model = onnx.load("E:\\apencpp\\x64\Debug\\best.onnx")
onnx.checker.check_model(model)
print("模型合法！")