@testset "kernels" begin
	R=Celerite2.RealKernel(0.5,1.0)
	C=Celerite2.ComplexKernel(1.0,0.6,6.0,3.0)
	SHO=Celerite2.SHOKernel(0.1, 2.0, -0.5)
	sum_2_kernels=R+SHO;		sum_3_kernels=R+SHO+C

	k1=celerite.RealTerm(0.5, 1.0) 
	k2=celerite.SHOTerm(0.1, 2.0, -0.5)
	k3=celerite.ComplexTerm(1.0,0.6,6.0,3.0)
	N,x,yerr,y=Celerite2.make_test_data()

	@test sum_2_kernels.kernels === KernelSum(R,SHO).kernels # === (R,SHO)
	@test Celerite2._get_coefficients(sum_2_kernels)==celerite.get_all_coefficients(k1+k2)
	@test Celerite2._get_coefficients(KernelSum(R,R))==celerite.get_all_coefficients(k1+k1)
	@test R(x) == celerite.get_value(k1,x)
	@test sum_2_kernels(x) == celerite.get_value(celerite.TermSum((k1,k2)),x)
	# @test sum_3_kernels(x)==celerite.get_value(celerite.TermSum((k1,k2,k3)),x) 

	prod_2_kernels = R*SHO ; prod_3_kernels=(R*SHO*C)
	@test Celerite2._get_coefficients(prod_2_kernels)==celerite.get_all_coefficients(k1*k2)

	  # check full covariance matrix 
	  function test_kernelmatrix(cel2,cel1)
		eps=1e-12
		res1=kernelmatrix(cel2,x,yerr) 
		res2=celerite._full_solve(cel1,x,y,yerr)[2] 
		 if  (maximum(abs.(res2.-res1)) < eps) 
		 	return true
		 else
			return false
			end
		end
		# @test test_kernelmatrix(SHO,k2) # did not test on TermSum because of different parametrization in gp.jl (doesn't have real terms separated) # test_kernelmatrix(C+D2,celerite.TermSum((k3,k2)))
  end