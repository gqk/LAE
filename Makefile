.EXPORT_ALL_VARIABLES:

CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0

BASE := "base/cifar100_order1.yaml"

vit_adapter:
	@nohup python main.py --config=$(@) extends="[${BASE}]" > $(@).out 2>&1 &

vit_lora:
	@nohup python main.py --config=$(@) extends="[${BASE}]" > $(@).out 2>&1 &

vit_prefix:
	@nohup python main.py --config=$(@) extends="[${BASE}]" > $(@).out 2>&1 &

swin_adapter:
	@nohup python main.py --config=$(@) extends="[${BASE}]" > $(@).out 2>&1 &

convnext_adapter:
	@nohup python main.py --config=$(@) extends="[${BASE}]" > $(@).out 2>&1 &
