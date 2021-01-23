#!/bin/bash
sudo apt-get install -y curl gnupg
curl -sfL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | apt-key add -
curl -sfL https://apt.repos.intel.com/setup/intelproducts.list -o /etc/apt/sources.list.d/intelproducts.list
sudo apt-get update
sudo apt-get install -y intel-mkl-2020.0.088
