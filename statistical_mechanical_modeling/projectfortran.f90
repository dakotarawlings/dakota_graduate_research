
subroutine CalcEnergy(Pos, Charge, M, L, rc,Lam0, epsilon,k_sigma, PEnergy, Dim, NAtom)    
    implicit none
    integer, intent(in) :: M, Dim, NAtom
    real(8), intent(in), dimension(0:NAtom-1, 0:Dim-1) :: Pos
    real(8), intent(in), dimension(0:NAtom-1) :: Charge
    real(8), intent(in) :: L, rc,Lam0, epsilon,k_sigma
    real(8), intent(out) :: PEnergy
    real(8), parameter :: k = 3000., r0 = 1., sigma0=cos(1.91)
    real(8), dimension(Dim):: rij, rjk,rji, Posi

    real(8) :: rc2,Shift,d, d2, id2, id6, id12, id3, id, sigma

    integer :: i,j
    PEnergy=0.
    
    Shift=-4.*(rc**(-12)-rc**(-6))
    rc2=rc*rc
    do i=0, NAtom-1
        Posi=pos(i,:)
        do j=i+1,NAtom-1
		
            if (j==i+1 .and. mod(j,M)>0) then
                !bonded interaction
                rij=Pos(j,:)-Posi
                rjk=Pos(j+1,:)-Pos(j,:)
                rij=rij-L*dnint(rij/L)
                
                d2=sum(rij*rij)
                d=d2**0.5

                PEnergy=PEnergy+(k/2.)*(d-r0)*(d-r0)	
		
            		if (mod(j+1,M)>0) then
            			!bond angle potential
                            	rjk=Pos(j+1,:)-Pos(j,:)
            			rjk=rjk-L*dnint(rjk/L)
            
                            	rji=Posi-Pos(j,:)
            			rji=rji-L*dnint(rji/L)
            
            			!sigma=acos(sum(rji*rjk)/(SQRT(sum(rji*rji))*SQRT(sum(rjk*rjk))))
				sigma=sum(rji*rjk)/(SQRT(sum(rji*rji))*SQRT(sum(rjk*rjk)))
            			PEnergy=PEnergy+k_sigma*(sigma-sigma0)
            		endif
		
            else
                !non-bonded interaction
                rij=Pos(j,:)-Posi
		
                rij=rij-L*dnint(rij/L)
                !d=sqrt(sum(rij*rij))
                d2=sum(rij*rij)
                d=d2**0.5
                if(d2>rc2) then
                   cycle
                end if
                id2=1./d2
                id6=id2*id2*id2
                id12=id6*id6
                PEnergy=PEnergy+4.*(id12-id6)+Shift+((Charge(i)*Charge(j))/(epsilon))*id*exp(-d/Lam0)
			

            endif
        enddo
    enddo
		
	
end subroutine



subroutine CalcEnergyForces(Pos, Charge, M, L, rc,Lam0, epsilon,k_sigma, PEnergy, Forces, Dim, NAtom)	
    implicit none
    integer, intent(in) :: M, Dim, NAtom
    real(8), intent(in), dimension(0:NAtom-1, 0:Dim-1) :: Pos
    real(8), intent(in), dimension(0:NAtom-1) :: Charge
    real(8), intent(in) :: L, rc,Lam0, epsilon,k_sigma
    real(8), intent(out) :: PEnergy
    real(8), intent(inout), dimension(0:NAtom-1, 0:Dim-1) :: Forces
!f2py intent(in, out) :: Forces
    real(8), parameter :: k = 3000., r0 = 1., sigma0=cos(1.91)
    real(8), dimension(Dim):: rij,rji, Fij,Fkangle,Fiangle, Posi, rjk
    
    real(8) :: rc2,Shift,d, d2,id, id2, id6, id12, id3,sigma

    integer :: i,j
    PEnergy=0.
    Forces=0.
    Shift=-4.*(rc**(-12)-rc**(-6))
    rc2=rc*rc
    do i=0, NAtom-1
        Posi=pos(i,:)
        do j=i+1,NAtom-1
		
            if (j==i+1 .and. mod(j,M)>0) then
                !bonded interaction
                rij=Pos(j,:)-Posi
                rij=rij-L*dnint(rij/L)

                d2=sum(rij*rij)
                d=d2**0.5

                PEnergy=PEnergy+(k/2.)*(d-r0)*(d-r0)	
                
                id=1./d
                Fij=rij*k*(1.-r0*id)

            		if (mod(j+1,M)>0) then
            			!bond angle potential

                            	rjk=Pos(j+1,:)-Pos(j,:)
            			rjk=rjk-L*dnint(rjk/L)
            
                            	rji=Posi-Pos(j,:)
            			rji=rji-L*dnint(rji/L)
            
            			!sigma=acos(sum(rji*rjk)/(SQRT(sum(rji*rji))*SQRT(sum(rjk*rjk))))
				sigma=sum(rji*rjk)/(SQRT(sum(rji*rji))*SQRT(sum(rjk*rjk)))
            			PEnergy=PEnergy+k_sigma*(sigma-sigma0)
            			
            
            			Fiangle=-(2*k_sigma*(sigma-sigma0))&
            			&*((rjk/(SQRT(sum(rji*rji))*SQRT(sum(rjk*rjk))))&
            			&-(sum(rji*rjk)/(((SQRT(sum(rji*rji)))**2)*SQRT(sum(rjk*rjk))))*(rji/SQRT(sum(rji*rji))))
            
            			Fkangle=-(2*k_sigma*(sigma-sigma0))&
            			&*((rji/(SQRT(sum(rji*rji))*SQRT(sum(rjk*rjk))))&
            			&-(sum(rji*rjk)/((SQRT(sum(rji*rji)))*(SQRT(sum(rjk*rjk)))**2))*(rjk/SQRT(sum(rjk*rjk))))
            
            			Forces(i,:)=Forces(i,:)+Fiangle
            			Forces(j,:)=Forces(j,:)-Fiangle-Fkangle
            			Forces(j+1,:)=Forces(j+1,:)+Fkangle
            			
                    	endif



                Forces(i,:)=Forces(i,:)+Fij
                Forces(j,:)=Forces(j,:)-Fij
		
            else
                !non-bonded interaction
                rij=Pos(j,:)-Posi
                rij=rij-L*dnint(rij/L)
                !d=sqrt(sum(rij*rij))
                d2=sum(rij*rij)
                if(d2>rc2) then
                   cycle
                end if
		d=d2**0.5
		id=1./d
		id3=id*id*id
                id2=1./d2
                id6=id2*id2*id2
                id12=id6*id6
                PEnergy=PEnergy+4.*(id12-id6)+Shift+((Charge(i)*Charge(j))/(epsilon))*id*exp(-d/Lam0)
                Fij=rij*((-48.*id12+24.*id6)*id2)+rij*(-((id3*Charge(i)*Charge(j))/(epsilon))&
                         &*exp(-d/Lam0)-((id2*Charge(i)*Charge(j))/(Lam0*epsilon))*exp(-d/Lam0))
                Forces(i,:)=Forces(i,:)+Fij
                Forces(j,:)=Forces(j,:)-Fij
		
            endif
        enddo
    enddo
		
	
end subroutine


subroutine VVIntegrate(Pos,Charge, Vel, Accel, M, L, rc,Lam0, epsilon,k_sigma, dt, KEnergy, PEnergy, Dim, NAtom)
    implicit none
    integer, intent(in) :: M, Dim, NAtom
    real(8), intent(in) :: L, rc,Lam0, epsilon,k_sigma, dt
    real(8), intent(inout), dimension(0:NAtom-1, 0:Dim-1) :: Pos, Vel, Accel
    real(8), intent(in), dimension(0:NAtom-1) :: Charge
!f2py intent(in,out) :: Pos, Vel, Accel
    real(8), intent(out) :: KEnergy, PEnergy
    external :: CalcEnergyForces
    Pos = Pos + dt * Vel + 0.5 * dt*dt * Accel
    Vel = Vel + 0.5 * dt * Accel
    call CalcEnergyForces(Pos, Charge,M, L, rc,Lam0, epsilon,k_sigma, PEnergy, Accel, Dim, NAtom)
    Vel = Vel + 0.5 * dt * Accel
    KEnergy = 0.5 * sum(Vel*Vel)
end subroutine


