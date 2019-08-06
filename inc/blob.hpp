#ifndef _BLOB_H_
#define _BLOB_H_

#include <vector>
#include <string>

#include "node.hpp"

using namespace std;

namespace framework {

class Blob : public Node {
public:
    Blob(void);
    Blob(string name);
    ~Blob(void);

private:
    vector<int> _dim;
};

}

#endif
