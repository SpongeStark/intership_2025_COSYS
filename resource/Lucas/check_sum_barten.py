def check_sum_barten():
    f_range = np.logspace(-1, 1.8, 1000)

    csf = get_barten_csf(f_range, 100)

    sigma_list, weight_list = get_weights_regression(f_range, get_barten_csf, dog_count=6)
    print("The sigmas are : ", sigma_list)
    print("The weights are : ", weight_list)

    graph = Graph("", "Frequency", "Sensitivity to contrast")
    graph.set_log_scale("xy")
    graph.set_ylim(0.01, 1000) # Pour avoir un graphique lisible

    graph.add_curve(f_range, csf, "CSF Barten")

    sum_slope_at_zero = np.zeros_like(f_range)
    for i, (sigma, weight) in enumerate(zip(sigma_list, weight_list)):
        slote_at_zero = get_slope_at_zero_crossing(sigma, f_range, weight)

        # Pour tracer chaque courbe individuellement :
        graph.add_curve(f_range, slote_at_zero, f"DoG {i + 1}", curve_type=CURVE)

        sum_slope_at_zero += slote_at_zero

    # Pour tracer la somme des DoG :
    graph.add_curve(f_range, sum_slope_at_zero, f"SDoG", curve_type=CURVE)

    graph.show()