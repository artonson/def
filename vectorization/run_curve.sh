for i in {0..55}
do
{ # try
    papermill parametric_curve_move_method.ipynb output_notebook/parametric_curve_move_method_${i}.ipynb -p id_it $i -p sharpness_threshold 0.06 -p cornerness_threshold 0.7 --execution-timeout 1800
    #save your output

} || { # catch
    echo "Error in : $i"
    # save log for exception 
}
done
