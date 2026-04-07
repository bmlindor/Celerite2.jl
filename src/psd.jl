    """
    Compute the real and positive roots of the PSD, which is given by:

    ∑_i (a_i z +b_i)/(z^2 + c_i*z + d_i)

    where z = ω^2. 
  """
  function _psd(kernel::CeleriteKernel, ω::AbstractVector)
  	  	# Compute the value of the power spectral density.
        ar, cr, ac, bc, cc, dc = _get_coefficients(kernel)
        ω² = ω.^2 
        ω⁴ = ω².^2 
        p = zeros(length(ω²))
        for i in 1:length(ar)
            p = p + ar[i]*cr[i] ./ (cr[i]*cr[i] .+ ω²)
        end
        for i in 1:length(ac)
            ω0² = cc[i]*cc[i]+dc[i]*dc[i]
            p = p .+ ((ac[i]*cc[i].+bc[i]*dc[i])*ω0².+(ac[i]*cc[i]-bc[i]*dc[i]).*ω²) ./ (ω⁴ + 2.0*(cc[i]*cc[i]-dc[i]*dc[i]).*ω².+ω0²*ω0²)
        end
        return sqrt(2.0 / pi) .* p
    end

	"""
	If we rewrite the power spectrum with z = ω^2, the condition for positive definiteness is met by:

	S(ω) = ∑_j (q_j * z + r_j ) / (z^2 + s_j * z + t_j) > 0

	Construct polynomial Q₀(z) = ∑_j (q_j * z + r_j ) Π_k (z^2 + s_k * z + t_k) = 0
	and its derivative. Evaluate coefficients of each polynomial in series. 

	"""
function _sturms_theorem(x::AbstractVector)
	# Compute coefficients in the numerator & denominator of the PSD.
	n_lor = round(Int, length(x)/4)

	# The order of the numerator polynomial in z = ω^2 that we need to
	# see if it has any zeros:
	pord = 2*(n_lor-1) + 1

	# Now, loop over coefficients:
	q = zeros(n_lor)
	r = zeros(n_lor)
	s = zeros(n_lor)
	t = zeros(n_lor)

	for i=1:n_lor
	  a_j = x[(i-1)*4+1]
	  b_j = x[(i-1)*4+2]
	  c_j = x[(i-1)*4+3]
	  d_j = x[(i-1)*4+4]
	  q[i] = a_j.*c_j.-b_j.*d_j
	  r[i] = (d_j.^2 .+c_j.^2).*(b_j.*d_j.+a_j.*c_j)
	  s[i] = 2 .*(c_j.^2 .-d_j.^2)
	  t[i] = (c_j.^2 .+d_j.^2).^2
	end

	# Initialize a polynomial:
	Q0 = Polynomial(zeros(pord+1))
	for i=1:n_lor
	# The polynomial for the current Lorentzian term in the common-denominator expression:
	  Qcur = Polynomial([r[i],q[i]])
	  for j=1:n_lor
	# Only multiply by the denominators from the other Lorentzians:
	    if j != i
	      Qcur *= Polynomial([t[j],s[j],1])
	    end
	  end
	  Q0 += Qcur
	end

	# Compute the roots of this polynomial:
	#poly_root = roots(Q0)

	# Now that we've computed coefficients of the polynomial, we just need
	# to apply Sturm's theorem!

	# Set up an array to hold signs of polynomial at zero & infinity:
	f_of_0 = zeros(pord+1)
	f_of_inf = zeros(pord+1)

	# Take the derivative of the polynomial:
	Q1 = derivative(Q0)
	#println(p0)
	#println(p1)
	# Insert the coefficients of the z^0 term:
	f_of_0[1] = Q0(0)
	f_of_0[2] = Q1(0)
	# Insert the coefficient of the z^(p_ord-i) term:
	f_of_inf[1] = Q0[pord]
	f_of_inf[2] = Q1[pord-1]

	# Now, loop over the Sturm polynomial series:
	for i=3:pord+1
	  Q2 = -rem(Q0,Q1)
	# Check that round-off error hasn't left us with a polynomial
	# that is the same order as Q0 or Q1, but with a small coefficient:
	  if length(Q2) >= length(Q1)
	    coeff = zeros(pord-i+2)
	    for j=0:pord-i+1
	      coeff[j+1]=Q2[j]
	    end
	    Q2 = Polynomial(coeff)
	  end
	# Insert the z^0 term:
	  f_of_0[i]=Q2(0)
	# Insert the z -> ∞ term:
	  f_of_inf[i]=Q2[pord-i+1]
	# Now move promote these polynomials, readying them for recursion in the next step:
	  Q0=copy(Q1)
	  Q1=copy(Q2)
	end

	# Now we'll compute the number of sign changes at z=0:
	sig_0 = 0
	for i=1:pord
	#  if f_of_0[i+1]*f_of_0[i] < 0 && abs(f_of_0[i+1]) > eps() && abs(f_of_0[i]) > eps()
	  if sign(f_of_0[i+1]) != sign(f_of_0[i])
	    sig_0 +=1
	  end
	end
	# Next, compute the number of sign changes at z=∞:
	sig_inf = 0
	for i=1:pord
	#  if f_of_inf[i+1]*f_of_inf[i] < 0 && abs(f_of_inf[i+1]) > eps() && abs(f_of_inf[i]) > eps()
	  if sign(f_of_inf[i+1]) != sign(f_of_inf[i])
	    sig_inf +=1
	  end
	end

	# The difference between z=0 & z=∞ gives the number of positive, real roots:
	#println("Number of positive, real roots: ",sig_0-sig_inf)

	# Print out the actual roots for inspection:
	#println("Roots: ",poly_root)

	# Now, determine which roots are positive & real from the numerial solver:
	n_pos_real = sig_0-sig_inf
	return n_pos_real
end
function _check_pos_roots(coeffs)
    ar, cr, ac, bc, cc, dc=coeffs
	aj = [ar;ac] ;cj = [cr;cc]; 
    bj = [zeros(length(ar));bc]; dj = [zeros(length(ar));dc]
    sturm_coeff = zeros(Float64,0)
    Jr = length(ar);    Jc = length(ac)
    for j=1:Jr+Jc
        push!(sturm_coeff,aj[j],bj[j],cj[j],dj[j])
    end
	# println("sturm_coeff: ",sturm_coeff)
    num_pos_root = _sturms_theorem(sturm_coeff)
	if num_pos_root > 0 
	  return false
	else
	return true
	end
end
