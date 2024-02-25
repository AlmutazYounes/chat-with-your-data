docker run -it \
   -e LOG=INFO \
   -p 8080:8080 \
   -p 8060:8060 \
   -p 8040:8040 \
   -v nucliadb-standalone:/data \
   nuclia/nucliadb:latest
