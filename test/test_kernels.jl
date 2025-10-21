@testset "kernels" begin
	R=Celerite2.RealKernel(log(0.5),log(1.0))
	C=Celerite2.ComplexKernel(log(1.0),log(0.6),log(6.0),log(3.0))
	S=Celerite2.SHOKernel(log(0.1), log(2.0), log(0.5))

	@test Celerite2.get_kernel(R) ==  [log(0.5),  log(1.0)]
	@test Celerite2.get_kernel(C) ==  [log(1.0),  log(0.6), log(6.0), log(3.0)]
	@test Celerite2.get_kernel(S) ==  [log(0.1),  log(2.0),log(0.5)]
	# @test Celerite2._get_coefficients(R) == ([0.5], [1.0], Float64[], Float64[], Float64[], Float64[])
	# @test Celerite2._get_coefficients(C) == (Float64[], Float64[], [1.0], [0.6], [6.0], [3.0000000000000004])

	@test isapprox(R(x[1:10][2:end]),[0.48652619, 0.45662424, 0.41771208, 0.38564919, 0.38558133,
       0.33454944, 0.32102731, 0.31308588, 0.29504727])
	@test isapprox(C(x[1:10][2:end]),[0.88766553, 0.65237196, 0.39647099, 0.23857257, 0.23829015,
       0.08234227, 0.0575695 , 0.04565173, 0.0248403 ])
	@test isapprox(round.(S(x[1:10])[2:end],sigdigits=7),[0.09999069, 0.09989785, 0.09960209, 0.09917618, 0.09917508,
       0.09805416, 0.09764373, 0.0973764 , 0.09668921])
	
	sum_2_kernels=R+C;		sum_3_kernels=R+S+C

	@test isapprox(sum_2_kernels(x[1:10])[2:end],[1.37419172, 1.1089962 , 0.81418306, 0.62422176, 0.62387148,
       0.41689172, 0.37859681, 0.35873761, 0.31988757])
	@test maximum(abs.(sum_3_kernels(x[1:10])[2:end] - [1.47418242, 1.20889405, 0.91378515, 0.72339794, 0.72304656,
       0.51494587, 0.47624054, 0.45611401, 0.41657678])) <= 1e-6

	prod_2_kernels = C*S

	@test maximum(abs.(prod_2_kernels(x[1:10])[2:end] .- [0.08875829, 0.06517056, 0.03948934, 0.02366072, 0.02363244,
       0.008074  , 0.0056213 , 0.0044454 , 0.00240179])) <= 1e-6
	# k1=celerite.RealTerm(log(0.5), log(1.0)) 
	# k2=celerite.SHOTerm(log(0.1), log(2.0), log(0.5))
	# k3=celerite.ComplexTerm(log(1.0),log(0.6),log(6.0),log(3.0))
	# @test Celerite2._get_coefficients(sum_2_kernels)==celerite.get_all_coefficients(k1+k2)
	# @test R(x) == celerite.get_value(k1,x)
	# @test sum_2_kernels(x) == celerite.get_value(celerite.TermSum((k1,k2)),x)
	# @test sum_3_kernels(x)==celerite.get_value(celerite.TermSum((k1,k2,k3)),x) 
	# @test Celerite2._get_coefficients(R*SHO)==celerite.get_all_coefficients(k1*k2)
  end