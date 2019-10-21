#echo "========= Inception V3 network =========="
#./nnc -g testsets/deploy_inception-v3.prototxt -o inception-v3.dot
#dot -Tpng inception-v3.dot -o inception-v3.png

echo "========= Inception V3 network (merged) =========="
 gdb --args ./nnc -g testsets/deploy_inception-v3-merge.prototxt -w testsets/inception-v3-merge.caffemodel -o inception-v3-merge.dot
#           ./nnc -g testsets/deploy_inception-v3-merge.prototxt -w testsets/inception-v3-merge.caffemodel -o inception-v3-merge.dot
dot -Tpng inception-v3-merge.dot -o inception-v3-merge.png

#echo "========= Lenet network        =========="
#./nnc -g testsets/lenet.prototxt -o lenet.dot -w testsets/lenet_iter_1000.caffemodel
#dot -Tpng lenet.dot -o lenet.png
