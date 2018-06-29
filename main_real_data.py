import merge_data as md
import numpy as np

# ----------------------------------------------------
# IMPORTING THE DATA
# ----------------------------------------------------

inputs, outputs, in_headers, out_headers = md.load_data()

print("===================")
print(in_headers)
print("-------------------")
print(inputs)
print("===================")
print(out_headers)
print("-------------------")
print(outputs)
print("===================")
print("Input size: ", inputs.shape)
print("Output size: ", outputs.shape)
print("===================")

# ----------------------------------------------------
# MANAGING THE DATA
# ----------------------------------------------------


