echo "========= Inception V3 network =========="
#./nnc -g testsets/deploy_inception-v3.prototxt -o output.dot
#dot -Tpng inceptionv1.dot -o inceptionv1.png
echo "========= Lenet network        =========="
./nnc -g testsets/lenet.prototxt -o lenet.dot
dot -Tpng lenet.dot -o lenet.png
