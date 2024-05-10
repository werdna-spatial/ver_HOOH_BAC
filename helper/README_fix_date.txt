find *.nc -maxdepth 1 -type f -exec  ncatted -O -a units,T,o,c,'seconds since 2000-01-01' {} \;
