
###The setup of the TORCS environment is completed thanks to Tolga Ok.

### Parts of the code are inspired and taken from https://github.com/jastfkjg/DDPG_Torcs_PyTorch

## Install

- **Install required libraries**

sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev  freeglut3-dev  libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev  libxrandr-dev libpng12-dev

- **Download repo**

git clone https://github.com/ugo-nama-kun/gym_torcs.git

- **cd into downloaded repo**

cd gym_torcs/vtorcs-RL-color/

./configure

- **below commands may require sudo**
make

make install

make datainstall


## Prepare

- **If you installed the game with sudo, you need to give your user account access to game files**

sudo chown -R YOUR_USERNAME /usr/local/share/games/torcs

- **Back to the main repo**

cd ..

- **Change the race configs**

mv quickrace.xml /usr/local/share/games/torcs/config/raceman/quickrace.xml


## How to Run

- **To train, run train.py**
- **To test, run test.py**
- **You can change the arguments in order to save specific models and run them**
