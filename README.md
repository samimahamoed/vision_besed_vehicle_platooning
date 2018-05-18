# vision_besed_vehicle_platooning
1. Get the vision-based code
```Shell
git clone https://github.com/samimahamoed/vision_besed_vehicle_platooning.git
cd vision_besed_vehicle_platooning
git checkout test
```
2. Get the submodule code
```Shell
git submodule init	# init the submodule
cd ES_Project_2018
git checkout master
# update submodule 
# check the version 26357aaf4d019b55997b84c97758a08f4d15434a
git pull
```

3. Run the code
```Shell
cd ..		# back to project root 
make run_py	# run the detection
# The detection show "No corners" if the checkerboard is not present
```
