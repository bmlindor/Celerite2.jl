using BenchmarkTools,Profile
using DelimitedFiles,CSV
    N1,x1,yerr1,y1 = make_test_data(100)
    N2,x2,yerr2,y2 = make_test_data(10000)
    N3,x3,yerr3,y3 = make_test_data(20000)
    N4,x4,yerr4,y4 = make_test_data(50000)
    N5,x5,yerr5,y5 = make_test_data(100000)
    N6,x6,yerr6,y6 = make_test_data(200000)
    N7,x7,yerr7,y7 = make_test_data(500000)
    N8,x8,yerr8,y8 = make_test_data(1000000)
    function time_model(x,yerr,y,N)
        println("timings for $N data points.")
        # non periodic component
        Q = 1.0/sqrt(2.0) ; w0 = 3.0
        S0 = var(y) ./ (w0 * Q)
        comp_1=SHOKernel(log(S0),log(Q),log(w0))

        # periodic component
        Q = 1.0; w0 = 3.0
        S0 = var(y) ./ (w0 * Q)
        comp_2=SHOKernel(log(S0),log(Q),log(w0))

        kernel = comp_1 + comp_2

        gp = CeleriteGP(kernel,x,yerr)

        coeff = _get_coefficients(gp.kernel)
        println("Factor.")
        # @show coeff
        logdetK0 = @time _factorize!(gp.D, gp.U, gp.W, gp.ϕ, coeff , x, gp.Σy)
        println("Solve.")
        invKy =  @time _solve!(gp.D, gp.U, gp.W, gp.ϕ, y)
        t = @benchmark logpdf($gp,$y)
        return t
    end

    # input = [x1,x2,x3,x4,x5,x6,x7,x8]
    # errors = [yerr1,yerr2,yerr3,yerr4,yerr5,yerr6,yerr7,yerr8]
    # data = [y1,y2,y3,y4,y5,y6,y7,y8]
    # Ns = [100,10000,20000,50000,100000,200000,500000,1000000]
    # ts = []
    # t = @benchmark time_model(input[$i],errors[end],data[end],Ns[end])

    # for i=1:length(input)
    #     t=time_model(input[i],errors[i],data[i],Ns[i])
    #     push!(ts,t)
    #     # Profile.print()  
    # end
	# comparing to DFM implementations
    Random.seed!(42)
    # dimensions of the problem
    maxN_power = 7 # 19 default
    minN_power = 6
    maxj_power = 2 # 8 default
    minj_power = 0

    N = 2 .^range(minN_power, stop= maxN_power+1)
    J = 2 .^range(minj_power, stop= maxj_power+1)
    filename="benchmark.txt"
    # make header
    # open(filename, "w+") do file
    #     for k in ["minN_power", "maxN_power", "minj_power", "maxj_power"]
    #         print(file,"# $k= ")
    #         println(file,getfield(Main, Symbol(k)))
    #         # write(file,)
    #     end
    #     println(file,"# N: $N")
    #     println(file,"# J: $J")
    #     println(file,"xi,yi,j,n,ll_time [μs]")#numpy_comp_time,numpy_ll_time
    # end

    @eval BenchmarkTools macro btimed(args...)
           _, params = prunekwargs(args...)
           bench, trial, result = gensym(), gensym(), gensym()
           trialmin, trialallocs = gensym(), gensym()
           tune_phase = hasevals(params) ? :() : :($BenchmarkTools.tune!($bench))
           return esc(quote
               local $bench = $BenchmarkTools.@benchmarkable $(args...)
               $BenchmarkTools.warmup($bench)
               $tune_phase
               local $trial, $result = $BenchmarkTools.run_result($bench)
               local $trialmin = $BenchmarkTools.minimum($trial)
               $result, $BenchmarkTools.time($trialmin)
           end)
       end
    # simulate dataset
    t = sort(randn(maximum(N)));
    yerr = rand(Uniform(0.1,0.2),length(t));
    y = sin.(t);

    # do_logLike(logdetK,n,y,inKy) = -0.5 *((logdetK + n * log(2*pi)) + (y' * invKy))
    compute(gp) = _factorize!(gp.D, gp.U, gp.W, gp.ϕ, _get_coefficients(gp.kernel) , collect(gp.x), gp.Σy)
    solve(gp,N) = _solve!(gp.D, gp.U, gp.W, gp.ϕ, y[1:N])
    # logpdf()
    function benchmark_pkg()

        for (xi,j) in enumerate(J)
            kernel = RealKernel(1.0, 0.1)
            for k in 1:((2*j - 1)%2)
                kernel += RealKernel(1.0,0.1)
            end
            for k in 1:((2*j - 1)/2)
                kernel += SHOKernel(0.1,2.0,1.6)#ComplexKernel(0.1,2.0,1.6,0.5)
            end
            # println("testing: ",kernel)

            for (yi,n) in enumerate(N)
                # y0 = y[1:n]
                local gp=CeleriteGP(kernel,t[1:n],yerr[1:n])
                # coeffs = 
                # Do cholesky decomposition and apply the inverse
                logdetK, comp_time = @btimed compute(gp)
                invK, inv_time = @btimed solve(gp,n)
                println("computation time: ",comp_time)
                println("inverse time: ",inv_time)
                # return @btime do_logLike(logdetK,n,y[1:n],invKy)#logpdf(gp,y[1:n])
                # println(xi," ",yi," ",coeffs)
                # println("$ll_time")
            # open(filename, "a+") do file
            #     write(file, "$xi, $yi, $j, $n, $ll_time \n")
            # end
            end
        end
    end
    benchmark_pkg()
    data=CSV.read(filename,header=7, DataFrame)
    plot(data.var" n"[1:15],[data.var" ll_time [μs]"][1:15] ./ 10^6)
    plot!(xscale=:log10, yscale=:log10, minorgrid=true)

        	# logL=logpdf(gp,y)


	# U,V,ϕ,A=_init_matrices(kernel,x,yerr)
	# logdetK = _factor_after_init!(A,U,V,ϕ)

	# evaluate the GP via cholesky factorization
	# N = length(x)

    # 	# Maximize the (log) marginal likelihood wrt. hyperparameters
	# vector = get_kernel(kernel)
	# mask = ones(Bool,size(kernel))
	# mask[2] = false 	# We don't want to fit the first Q
