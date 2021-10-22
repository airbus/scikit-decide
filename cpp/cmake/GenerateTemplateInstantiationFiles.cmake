# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This function generates .cc template instantion files from a given
# template file (the latter "template" word is not in the sense of
# C++ templates but CMake configuration files) for each possible
# combination of C++ template instantiations given as arguments
# First argument: CMake "template file"
# Second argument: Output directory for the generated files
# Other arguments: lists of pairs of C++ template instantiations
# A pair is in the form "T" "ClassSubstituation1;ClassSubstituation2;..."
# where T is the template to be successively substituted to
# ClassSubstituation1; then ClassSubstituation2; then etc.
# Each substitution can optionally end with "!short_name" where
# short_name is the substitution name that will be part to the
# generated template instantiation file.
# The generated C++ template instantiation files are stored in
# the first variable of the function
# Example usage:
# generate_template_instantion_files(generated_template_instantiation_files
#                                    "myfile.cc.in"
#                                    "myoutputdir"
#                                    "T1" "MyClass1!Cl1;MyClass2"
#                                    "T2" "MyClass3;MyClass4!Cl4")
# This call will generate the following files that will be recorded in
# the variable generated_template_instantiation_files:
# myoutputdir/myfile_Cl1MyClass3.cc with T1 replaced with MyClass1 and T2 with MyClass3 in myfile.cc.in
# myoutputdir/myfile_Cl1Cl4.cc with T1 replaced with MyClass1 and T2 with MyClass4 in myfile.cc.in
# myoutputdir/myfile_MyClass2MyClass3.cc with T1 replaced with MyClass2 and T2 with MyClass3 in myfile.cc.in
# myoutputdir/myfile_MyClass2MyCl4.cc with T1 replaced with MyClass2 and T2 with MyClass4 in myfile.cc.in

SET_PROPERTY(GLOBAL PROPERTY GENERATED_TEMPLATE_INSTANTIATION_FILES)

FUNCTION(generate_template_instantiation_files_recursive)
    IF("${ARGV0}" STREQUAL "${ARGV1}")  # end of recursion
        LIST(SUBLIST ARGN 2 -1 CURRENT_INSTANTIATIONS)
        STRING(FIND ${GTIF_TEMPLATE_FILE} "/" filename_begin REVERSE)
        MATH(EXPR filename_begin "${filename_begin} + 1" OUTPUT_FORMAT DECIMAL)
        STRING(SUBSTRING ${GTIF_TEMPLATE_FILE} ${filename_begin} -1 filename_substr)
        STRING(FIND ${filename_substr} ".cc.in" filename_insertion_point REVERSE)
        STRING(SUBSTRING ${filename_substr} 0 ${filename_insertion_point} INSTANTIATION_FILENAME)
        SET(INSTANTIATION_FILENAME "${INSTANTIATION_FILENAME}_")
        LIST(LENGTH GTIF_TEMPLATE_NAMES NUMBER_TEMPLATES)
        MATH(EXPR TEMPLATE_RANGE "${NUMBER_TEMPLATES} - 1")

        FOREACH(template_index RANGE 0 ${TEMPLATE_RANGE})
            LIST(GET GTIF_TEMPLATE_NAMES ${template_index} template_name)
            LIST(GET CURRENT_INSTANTIATIONS ${template_index} class_name)
            STRING(FIND ${class_name} "!" exclamation_mark_pos REVERSE)

            IF ("${exclamation_mark_pos}" STREQUAL "-1")
                SET(short_class_name_begin 0)
            ELSE()
                MATH(EXPR short_class_name_begin "${exclamation_mark_pos} + 1" OUTPUT_FORMAT DECIMAL)
            ENDIF()

            STRING(SUBSTRING ${class_name} ${short_class_name_begin} -1 short_class_name)
            STRING(SUBSTRING ${class_name} 0 ${exclamation_mark_pos} class_name)
            SET(INSTANTIATION_FILENAME "${INSTANTIATION_FILENAME}${short_class_name}")
            SET(${template_name} ${class_name})
        ENDFOREACH()

        SET(INSTANTIATION_FILENAME "${INSTANTIATION_FILENAME}.cc")
        CONFIGURE_FILE("${GTIF_TEMPLATE_FILE}" "${GTIF_OUTPUT_DIR}/${INSTANTIATION_FILENAME}")
        IF("${GTIF_TEMPLATE_INSTANTIATION_FILES}" STREQUAL "")
            SET(GTIF_TEMPLATE_INSTANTIATION_FILES "${GTIF_OUTPUT_DIR}/${INSTANTIATION_FILENAME}" PARENT_SCOPE)
        ELSE()
            SET(GTIF_TEMPLATE_INSTANTIATION_FILES
                "${GTIF_TEMPLATE_INSTANTIATION_FILES};${GTIF_OUTPUT_DIR}/${INSTANTIATION_FILENAME}"
                PARENT_SCOPE)
        ENDIF()
    ELSE()
        LIST(GET GTIF_TEMPLATE_NAMES ${ARGV0} template_name)
        FOREACH(class_name IN LISTS GTIF_CLASS_NAMES_${template_name})
            LIST(SUBLIST ARGN 2 -1 CURRENT_INSTANTIATIONS)
            LIST(APPEND CURRENT_INSTANTIATIONS ${class_name})
            MATH(EXPR NEW_INDEX "${ARGV0} + 1" OUTPUT_FORMAT DECIMAL)
            generate_template_instantiation_files_recursive(${NEW_INDEX} ${ARGV1} ${CURRENT_INSTANTIATIONS})
            SET(GTIF_TEMPLATE_INSTANTIATION_FILES "${GTIF_TEMPLATE_INSTANTIATION_FILES}" PARENT_SCOPE)
        ENDFOREACH()
    ENDIF()
ENDFUNCTION()

FUNCTION(generate_template_instantiation_files)
    IF(${ARGC} LESS 5)
        MESSAGE(FATAL_ERROR "Function generate_template_instantiation_files expects at least 5 arguments")
    ENDIF()

    SET(GTIF_TEMPLATE_FILE ${ARGV1})
    SET(GTIF_OUTPUT_DIR ${ARGV2})
    MATH(EXPR ARGS_REMAINDER "${ARGC} % 2" OUTPUT_FORMAT DECIMAL)

    IF(NOT ${ARGS_REMAINDER} STREQUAL "1")
        MESSAGE(FATAL_ERROR "Function generate_template_instantiation_files expects an odd number of parameters.")
    ENDIF()

    SET(GTIF_TEMPLATE_NAMES)
    MATH(EXPR ARGC_BY_TWO "((${ARGC} - 1) / 2) - 1" OUTPUT_FORMAT DECIMAL)

    FOREACH(index RANGE 1 ${ARGC_BY_TWO})
        MATH(EXPR odd_index "(${index} * 2) + 1" OUTPUT_FORMAT DECIMAL)
        SET(ODD_ARG ${ARGV${odd_index}})
        LIST(APPEND GTIF_TEMPLATE_NAMES ${ODD_ARG})
        MATH(EXPR even_index "(${index} * 2) + 2" OUTPUT_FORMAT DECIMAL)
        SET(EVEN_ARG ${ARGV${even_index}})
        SET(GTIF_CLASS_NAMES_${ODD_ARG} "${EVEN_ARG}")
    ENDFOREACH()

    SET(GTIF_TEMPLATE_INSTANTIATION_FILES "")
    generate_template_instantiation_files_recursive(0 ${ARGC_BY_TWO} "")
    SET(${ARGV0} "${GTIF_TEMPLATE_INSTANTIATION_FILES}" PARENT_SCOPE)
ENDFUNCTION()
