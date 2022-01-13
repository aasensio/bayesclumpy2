module bcMod
use iso_c_binding, only: c_int, c_float

implicit none

	type full_database_type
		integer :: nY, nsig, nN0, nq, ntauv, ni, nlambda, db_id
		integer :: ny_id, nsig_id, nn0_id, nq_id, ntauv_id, ni_id, npca_id, nlam_id
		integer :: lam_id, coefs_id, base_id, meansed_id
		integer :: y_id, sig_id, n0_id, q_id, tauv_id, i_id, npca, nlam
    	real(kind=4), pointer :: Y(:), sig(:), N0(:), q(:), tauv(:), i(:)
      	real(kind=4), pointer :: coefs(:,:,:,:,:,:,:), base(:,:)
      	real(kind=4), pointer :: meanSed(:), lambda(:), wrk(:,:), params(:,:), SED_highres(:)
      	integer, pointer :: indi(:), ee(:,:)
	end type full_database_type

	type(full_database_type) :: database

contains
	
	subroutine c_initialize_database(nY, nsigma, nn0, nq, ntauv, ni, npca, nlam, Y, sigma, n0, q, tauv, ii, lambda, base, coefs, meanSED) bind(c)
	integer(c_int) :: nY, nsigma, nn0, nq, ntauv, ni, npca, nlam
	real(c_float), intent(in) :: Y(nY), sigma(nsigma), n0(nn0), q(nq), tauv(ntauv), ii(ni), lambda(nlam), base(npca, nlam), meansed(nlam)
	real(c_float), intent(in) :: coefs(npca,ni,ntauv,nq,nn0,nsigma,ny)

	integer :: ndim, i, j, k, nr, c1, c2, nmax
		
		database%nY = nY
		database%nn0 = nn0
		database%nq = nq
		database%ntauv = ntauv
		database%nsig = nsigma
		database%ni = ni
		database%npca = npca
		database%nlam = nlam
		
		allocate(database%lambda(database%nlam))
		allocate(database%base(database%npca,database%nlam))
		allocate(database%coefs(database%npca,database%ni,database%ntauv,database%nq,database%nn0,database%nsig,database%ny))
		allocate(database%meansed(database%nlam))	
		allocate(database%y(database%ny))
		allocate(database%sig(database%nsig))
		allocate(database%n0(database%nn0))
		allocate(database%q(database%nq))
		allocate(database%tauv(database%ntauv))
		allocate(database%i(database%ni))
		allocate(database%wrk(database%npca,2**6))
		allocate(database%SED_highres(database%nlam))
		
		database%Y = Y
		database%sig = sigma
		database%n0	= n0
		database%q = q
		database%tauv = tauv
		database%i = ii
		database%lambda = lambda
		database%base = base
		database%coefs = coefs
		database%meanSED = meanSED
		
	! Do some precomputations for accelerating the linear interpolation routines on the database
		ndim = 6
		allocate(database%indi(ndim))
		do i = 1, ndim
			database%indi(i) = ndim - i + 1
		enddo

		allocate(database%ee(ndim,2**ndim))
		do i = 1, ndim
			nr = 2**(i-1)
			c1 = 1
			do j = 1, nr
				c2 = 1
				do k = 1, 2**ndim/nr
					database%ee(ndim-database%indi(i)+1,c1) = (c2-1) / 2**(ndim-i)
					c1 = c1 + 1
					c2 = c2 + 1
				enddo
			enddo
		enddo

	! Make array of parameters
		nmax = maxval( (/database%nY,database%nsig,database%nn0,database%nq,database%ntauv,database%ni/) )
		allocate(database%params(6,nmax))

		database%params = 1.e10

		database%params(1,1:database%nY) = database%Y
		database%params(2,1:database%nsig) = database%sig
		database%params(3,1:database%nn0) = database%n0
		database%params(4,1:database%nq) = database%q
		database%params(5,1:database%ntauv) = database%tauv
		database%params(6,1:database%ni) = database%i

	end subroutine c_initialize_database


	subroutine c_lininterpol_database(pars, coefs) bind(c)
	real(c_float), intent(in) :: pars(6)
	real(c_float), intent(out) :: coefs(database%npca)
	real(c_float) :: delta
	integer :: i, j, ndim, near(6), indices(6), ind
		
		ndim = 6

! Find the indices of the hypercube around the desired value
		call hunt(database%Y,database%nY,pars(1),near(1))
		call hunt(database%sig,database%nsig,pars(2),near(2))
		call hunt(database%n0,database%nn0,pars(3),near(3))
		call hunt(database%q,database%nq,pars(4),near(4))
		call hunt(database%tauv,database%ntauv,pars(5),near(5))
		call hunt(database%i,database%ni,pars(6),near(6))
		
		
! Extract the values of the function that will be used
		do i = 1, 2**ndim
			indices = near + database%ee(:,i)
			database%wrk(:,i) = database%coefs(:,indices(6),indices(5),indices(4),indices(3),indices(2),indices(1))			
		enddo
		
! Do the actual linear interpolation
		do i = 1, ndim
			ind = database%indi(i)
			
			delta = -(pars(ind) - database%params(ind,near(ind))) / &
				(database%params(ind,near(ind)) - database%params(ind,near(ind)+1))
				
			do j = 1, 2**(ndim-i)
				
				database%wrk(:,j) = (database%wrk(:,2*j) - database%wrk(:,2*j-1)) * delta + database%wrk(:,2*j-1)
				
			enddo
			
		enddo
		
		coefs = database%wrk(:,1)		
		
	end subroutine c_lininterpol_database

	! ---------------------------------------------------------
!	Given an array xx(1:n), and given a value x, returns a value jlo such that x is between
!	xx(jlo) and xx(jlo+1). xx(1:n) must be monotonic, either increasing or decreasing.
!	jlo=0 or jlo=n is returned to indicate that x is out of range. jlo on input is taken as
!	the initial guess for jlo on output.
! ---------------------------------------------------------
	subroutine hunt(xx,n,x,jlo)
	integer :: jlo,n
	real(kind=4) :: x,xx(n)
	integer :: inc,jhi,jm
	logical :: ascnd

		ascnd=xx(n).ge.xx(1)
		if (jlo.le.0.or.jlo.gt.n) then
			jlo=0
			jhi=n+1
			goto 3
		endif
		inc=1
		if (x.ge.xx(jlo).eqv.ascnd) then
1     	jhi=jlo+inc
			if (jhi.gt.n) then
				jhi=n+1
			else if (x.ge.xx(jhi).eqv.ascnd) then
				jlo=jhi
				inc=inc+inc
				goto 1
			endif
		else
			jhi=jlo
2     	jlo=jhi-inc
			if (jlo.lt.1) then
				jlo=0
			else if (x.lt.xx(jlo).eqv.ascnd) then
				jhi=jlo
				inc=inc+inc
				goto 2
			endif
		endif
3 		if (jhi-jlo.eq.1) then
			if(x.eq.xx(n)) jlo=n-1
			if(x.eq.xx(1)) jlo=1
			return
		endif
		jm = (jhi+jlo)/2
		if (x.ge.xx(jm).eqv.ascnd) then
			jlo=jm
		else
			jhi=jm
		endif
		goto 3
	end subroutine hunt


end module bcMod
