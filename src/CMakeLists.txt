#
################################################################
#           Configure and Install PolyCalc Library
################################################################
#
# We do this from inside the "src" directory so that the include
# installation is not corrupted with all the temporary cmake file
# which will now be placed inside "src"
#  
 
#
# Define the name of the xstd Library
#
set(polycalc_library_name ${PROJECT_NAME})

#
# Make a header only library target
#
add_library(${polycalc_library_name} INTERFACE)

#
# List the include directories for tests and library users
#
target_include_directories(${polycalc_library_name} 
	INTERFACE
	    "$<BUILD_INTERFACE:${MY_PROJECT_INCDIR}>"
    	"$<INSTALL_INTERFACE:${MY_INSTALL_INCDIR}>"
)

#
# List of all libraries that need linking
#
# target_link_libraries(${xstd_library_name} 
# 	INTERFACE
# 	    "$<$<BOOL:${BLAS_FOUND}>:${BLAS_LIBRARIES}>"
# 		"$<$<BOOL:${CBLAS_FOUND}>:${CBLAS_LIBRARIES}>"
#    	"$<$<BOOL:${BLIS_FOUND}>:${BLIS_LIBRARIES}>"
# )

set_target_properties(${polycalc_library_name}
	PROPERTIES
        CXX_STANDARD 17
    	CXX_STANDARD_REQUIRED YES
    	CXX_EXTENSIONS NO
)

#
# Copy all header files to the installation folder
#
install(FILES "${MY_PROJECT_INCDIR}/xstd.hpp" DESTINATION ${MY_INSTALL_INCDIR})
install(DIRECTORY "${MY_PROJECT_INCDIR}/xstd" DESTINATION ${MY_INSTALL_INCDIR})