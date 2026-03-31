# NOTE:
The model's crashed when trying to generate stories because of exploded gradients.
Therefore there is no generate stories csv


### Error
- the model's weights exploded to NaN (exploding gradients)

### fix in v1.4
- replaced float16 with bfloat16
- reduced learning rate to 0.00005
