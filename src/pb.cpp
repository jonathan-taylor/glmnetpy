#include <pybind11/embed.h>

void update_pb(py::object pb, int step_inc) {
    pb.attr("update")(step_inc);
}
