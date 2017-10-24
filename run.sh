#!/bin/bash
#
# Copyright [2013-2014] PayPal Software Foundation
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

mvn clean install -DskipTests

scp -P 3003 target/shifu-0.2.8-SNAPSHOT-hdp-yarn.tar.gz localhost:~/shifu/shifu-0.2.8

ssh -p 3003 localhost '~/shifu/shifu-0.2.8/run_dt.sh'