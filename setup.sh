#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install python3-pip python-pip

sudo pip3 install -U pip
sudo pip3 install -r ${cur_dir}/requirements.txt
