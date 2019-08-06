./nnc -g testsets/deploy_inception-v3.prototxt -o output.dot
dot -Tpng output.dot -o output.png
