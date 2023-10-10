Frequently asked questions
===================================

Q: I use libjpeg-turbo with the float DCT. Why do the outputs of jpeglib sometimes differ from cjpeg?

A: It is [known and documented](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/README.md#mathematical-compatibility) that the outputs of libjpeg-turbo can deviate from libjpeg 6b. Please check whether cjpeg was compiled in the same way as the libjpeg-turbo included in your jpeglib package.