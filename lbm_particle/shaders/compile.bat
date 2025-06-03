@echo off
glslangValidator particle.vert -V -o particle_vert.spv
glslangValidator particle.frag -V -o particle_frag.spv
glslangValidator wireframe.vert -V -o wireframe_vert.spv
glslangValidator wireframe.frag -V -o wireframe_frag.spv
glslangValidator calc.comp -V -o calc_comp.spv
glslangValidator collide_and_stream.comp -V -o collide_and_stream_comp.spv
glslangValidator update_macro.comp -V -o update_macro_comp.spv
glslangValidator apply_bc.comp -V -o apply_bc_comp.spv