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

    template<typename First, typename... Args>
    void set_dim(First first, Args... args);
    void set_dim(int arg);
    void set_dim(std::vector<int> dim);

private:
    vector<int> _dim;
};

}

#endif
