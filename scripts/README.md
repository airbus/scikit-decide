This file host the docker file to build scikit-decide on manylinux platform

```
> docker build -t skdecide_x86-64 --output type=local,dest=tmpwheelhouse -f scripts/Dockerfile_x86_64 .
> docker run -it skdecide_x86-64 /bin/bash
