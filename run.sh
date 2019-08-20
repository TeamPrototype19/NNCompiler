#echo "========= Inception V3 network =========="
#./nnc -g testsets/deploy_inception-v3.prototxt -o inception-v3.dot
#dot -Tpng inception-v3.dot -o inception-v3.png
echo "========= Lenet network        =========="
./nnc -g testsets/lenet.prototxt -o lenet.dot -w testsets/lenet_iter_1000.caffemodel
dot -Tpng lenet.dot -o lenet.png
