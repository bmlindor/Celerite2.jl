@testset "kernels" begin
	R=Celerite2.RealKernel(log(0.5),log(1.0))
	C=Celerite2.ComplexKernel(log(1.0),log(0.6),log(6.0),log(3.0))
	S=Celerite2.SHOKernel(log(0.1), log(2.0), log(0.5))

	@test get_kernel(R) ==  [log(0.5),  log(1.0)]
	@test get_kernel(C) ==  [log(1.0),  log(0.6), log(6.0), log(3.0)]
	@test get_kernel(S) ==  [log(0.1),  log(2.0),log(0.5)]

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

	# alt_SHO=celerite.RealTerm(log(0.5), log(1.0)) 
	# standard_SHO=celerite.SHOTerm(log(0.1), log(2.0), log(0.5))
	# alt_SHO_2=celerite.ComplexTerm(log(1.0),log(0.6),log(6.0),log(3.0))
	# @test Celerite2._get_coefficients(sum_2_kernels)==celerite.get_all_coefficients(alt_SHO+standard_SHO)
	# @test R(x) == celerite.get_value(alt_SHO,x)
	# @test sum_2_kernels(x) == celerite.get_value(celerite.TermSum((alt_SHO,standard_SHO)),x)
	# @test sum_3_kernels(x)==celerite.get_value(celerite.TermSum((alt_SHO,standard_SHO,alt_SHO_2)),x) 
	# @test Celerite2._get_coefficients(R*SHO)==celerite.get_all_coefficients(alt_SHO*standard_SHO)
  end
  @testset "kwarg kernels" begin
	alt_SHO = SHOKernel(;log_σ = 1.0,log_Q = 4.0, log_ρ = 10.0)
	standard_SHO = SHOKernel(;log_S0 = 6.1621229335906555,log_Q = 4.0, log_ω0 = -8.162122933590656)
	alt_SHO_2 = SHOKernel(;log_σ = 1.0,log_Q = 4.0, log_ω0 =  -8.162122933590656)
	alt_SHO_3 = SHOKernel(;log_σ = 1.0,log_τ = 12.855270114150601, log_ρ = 10.0)

	@test alt_SHO_3.log_ρ == standard_SHO.log_ρ == alt_SHO.log_ρ == alt_SHO_2.log_ρ
	@test alt_SHO_3.log_ω0 == standard_SHO.log_ω0 == alt_SHO.log_ω0 == alt_SHO_2.log_ω0
	@test alt_SHO_3.log_S0 == standard_SHO.log_S0 == alt_SHO.log_S0 == alt_SHO_2.log_S0
	@test alt_SHO_3.log_τ == standard_SHO.log_τ == alt_SHO.log_τ == alt_SHO_2.log_τ
	@test alt_SHO_3.log_σ == standard_SHO.log_σ == alt_SHO.log_σ == alt_SHO_2.log_σ
	@test alt_SHO_3.log_Q == standard_SHO.log_Q == alt_SHO.log_Q == alt_SHO_2.log_Q

	@test get_kernel(alt_SHO) == get_kernel(standard_SHO) == get_kernel(alt_SHO_2) == get_kernel(alt_SHO_3)

	kwarg_SHO = SHOKernel(;log_S0 = 0.1,log_ω0 = 2.0, log_Q = 0.5)

	set_kernel!(alt_SHO_3,get_kernel(kwarg_SHO))
	
	@test alt_SHO_3.log_ρ == kwarg_SHO.log_ρ 
	@test alt_SHO_3.log_ω0 == kwarg_SHO.log_ω0 
	@test alt_SHO_3.log_S0 == kwarg_SHO.log_S0 
	@test alt_SHO_3.log_τ == kwarg_SHO.log_τ
	@test alt_SHO_3.log_σ == kwarg_SHO.log_σ
	@test alt_SHO_3.log_Q == kwarg_SHO.log_Q 
  end