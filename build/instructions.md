# Making a conda installable package

## Build
First, please update the release version and all meta info in 

    - setup.py
    - README.md
    - conda recipe
    - more??

This is to be replaced by a more automatic process in the future but for now we still have to do it manually.

## Automatic build
Use the `build/automate_build.sh` script after carefully checking all the script variables. The build directory must be empty on running the script.

    
## Checking the build   
 
After a successful build, conda shows the path to the build package. Alternatively, type to find channels:

    conda config --show
    
Build index so that conda can find the package:

    conda index --verbose <channel>
    
Check if it is locally available:

    conda search -c local -i -v <pkg_name>
    
The last two steps can be skipped if you build for a different machine.

To remove package from channel, delete the package tarball and update the index:

    conda index --verbose <CHANNEL>

## Usage in different environment (user space)
The packages tar-file can be delivered. For now, we publish it manually on our self-hosted channel `CHANNEL`. The package has to go into the `noarch/` subdirectory.

Place the package there:

    CHANNEL=<dir>
    CHANNEL/noarch/<pkg_name>
       
Index the channel

    conda index $CHANNEL
    
(Optional) Check that it is found by conda:

    conda search -c CHANNEL <pkg_name>
    
Do the usual conda install but you must indicate the local channel:

    conda install -c CHANNEL siamese
    