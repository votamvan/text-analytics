!apt install git git-lfs vim tree zip unzip
import os
os.chdir('/content/')
!test -d /content/text-analytics || rm -rf /content/text-analytics
!git clone https://github.com/votamvan/text-analytics.git
os.chdir('/content/text-analytics/')
!git pull && git lfs pull
!cat resources.zipa* > resources.zip
!unzip resources.zip && rm -f resources.zip*