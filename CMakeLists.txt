cmake_minimum_required(VERSION  3.14.5)
project(test)
find_package(OpenCV REQUIRED)


set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES
#        main.cpp
#        main1.cpp
#        main2.cpp
#        main3.cpp
#        main4.cpp
#        main5.cpp
#        main6.cpp
#        main8.cpp
#        main9.cpp
#        main910.cpp
#        main11.cpp
#        main12.cpp
#        main13.cpp
#        main14.cpp
        main15.cpp
        )


add_executable(test ${SOURCE_FILES})



target_link_libraries(test ${OpenCV_LIBS})


