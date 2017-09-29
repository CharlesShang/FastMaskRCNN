#https://stackoverflow.com/documentation/tensorflow/3883/how-to-debug-a-memory-leak-in-tensorflow#t=201612280142239281993

#To improve memory allocation performance, many TensorFlow users often use tcmalloc instead of the default malloc() implementation, as tcmalloc suffers less from fragmentation when allocating and deallocating #large objects (such as many tensors). Some memory-intensive TensorFlow programs have been known to leak heap address space (while freeing all of the individual objects they use) with the default malloc(), but #performed just fine after switching to tcmalloc. In addition, tcmalloc includes a heap profiler, which makes it possible to track down where any remaining leaks might have occurred.

#The installation for tcmalloc will depend on your operating system, but the following works on Ubuntu 14.04 (trusty) (where script.py is the name of your TensorFlow Python program):

#sudo apt-get install google-perftools4
LD_PRELOAD=/usr/lib/libtcmalloc.so.4 python train/train.py

#As noted above, simply switching to tcmalloc can fix a lot of apparent leaks. However, if the memory usage is still growing, you can use the heap profiler as follows:

#LD_PRELOAD=/usr/lib/libtcmalloc.so.4 HEAPPROFILE=/tmp/profile python script.py ...
#After you run the above command, the program will periodically write profiles to the filesystem. The sequence of profiles will be named:

#/tmp/profile.0000.heap
#/tmp/profile.0001.heap
#/tmp/profile.0002.heap
#...
#You can read the profiles using the google-pprof tool, which (for example, on Ubuntu 14.04) can be installed as part of the google-perftools package. For example, to look at the third snapshot collected above:

#google-pprof --gv `which python` /tmp/profile.0002.heap
#Running the above command will pop up a GraphViz window, showing the profile information as a directed graph.