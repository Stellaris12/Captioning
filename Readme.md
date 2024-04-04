This is a modified version of wdv3-jax by SmilingWolf that taggs images in a folder and allows the user to delete images based on their tags.
to install
#Windows Install Instructions 
# Create VENV
python -m venv .venv
# Activate it
source .venvscriptsactivate
# Upgrade pipsetuptoolswheel
python -m pip install -U pip setuptools wheel
# At this point, optionally you can install JAX manually (e.g. if you are using an nVidia GPU)
python -m pip install -U jax[cpu]
# Install requirements
python -m pip install -r requirements.txt
# Launch
python wdv3_jax.py --model swinv2 test.jpg`

Included is the tag analisys script to help manage already tagged datasets. 

#Known Issues 
Only Processes one image at a time 
Reloads the model on every itteration
Requires a image to launch
