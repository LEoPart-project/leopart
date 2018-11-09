#### Forms theta formulation, do not throw away!!
#v, q = TestFunctions(mixedL)
    #u, p = TrialFunctions(mixedL)

    #vbar, qbar = TestFunctions(mixedG)
    #ubar, pbar = TrialFunctions(mixedG)

    ## Penalty
    #alpha = Constant(6*k*k)
    #beta  = Constant(0.)

    ## Mesh related
    #n  = FacetNormal(mesh)
    #he = CellDiameter(mesh)

    #pI = p*Identity(V.cell().topological_dimension())
    #pbI= pbar*Identity(V.cell().topological_dimension())

    #p0I = p0*Identity(V.cell().topological_dimension())
    #pb0I= pbar0*Identity(V.cell().topological_dimension())

    ##
    ## Get it in block form:
    ##
    ## | A   G |
    ## |       | V = R
    ## | G^T B |

    ## Theta scheme
    #AB = dot(u,v)/dt * dx \
        #+ theta * inner( 2*nu*sym(grad(u)),grad(v) )*dx \
        #+ theta * facet_integral( dot(-2*nu*sym(grad(u))*n + (2*nu*alpha/he)*u,v) ) \
        #+ theta*facet_integral( dot(-2*nu*u,sym(grad(v))*n) ) \
        #- theta * inner(pI,grad(v))*dx
    #BtF= -dot(q,div(u))*dx - facet_integral(beta*he/(nu+1)*dot(p,q))
    #A  = AB + BtF
    #A  = Form(A)

    ## Upper right block G
    #CD= theta   * facet_integral(-alpha/he*2*nu*inner( ubar,v ) ) \
        #+ theta * facet_integral( 2*nu*inner(ubar, sym(grad(v))*n) ) \
        #+ theta * facet_integral(dot(pbI*n,v))
    #H = facet_integral(beta*he/(nu+1)*dot(pbar,q))

    #G = CD + H
    #G = Form(G)

    #CDT = theta * facet_integral(- alpha/he*2*nu*inner( vbar, u ) ) \
        #+ theta * facet_integral( 2*nu*inner(vbar, sym(grad(u))*n) ) \
        #+ facet_integral( qbar * dot(u,n))
    #HT   = theta * facet_integral(beta*he/(nu+1)*dot(p,qbar))

    #GT   = CDT + HT
    #GT   = Form(GT)

    ## Lower right block B
    #KL = theta * facet_integral( alpha/he * 2*nu*dot(ubar,vbar)) - theta * facet_integral( dot(pbar*n,vbar) )
    #LtP= -facet_integral(dot(ubar,n)*qbar) - facet_integral( beta*he/(nu+1) * pbar * qbar )
    #B = KL + LtP
    #B = Form(B)

    ##Righthandside
    #Q = dot(f,v)*dx + dot(u0,v)/dt * dx \
        #- (1-theta) * inner( 2*nu*sym(grad(u0)),grad(v) )*dx \
        #- (1-theta) * facet_integral( dot(-2*nu*sym(grad(u0))*n + (2*nu*alpha/he)*u0,v) ) \
        #- (1-theta) * facet_integral( dot(-2*nu*u0,sym(grad(v))*n) ) \
        #+ (1-theta) * inner(p0I,grad(v))*dx \
        #- (1-theta) * facet_integral(-alpha/he*2*nu*inner( ubar0,v ) ) \
        #- (1-theta) * facet_integral( 2*nu*inner(ubar0, sym(grad(v))*n) ) \
        #- (1-theta) * facet_integral(dot(pb0I*n,v))
    #S = facet_integral( dot( Constant((0,0)), vbar) ) #\
        ##- (1-theta) * facet_integral(-alpha/he*2*nu*inner( u0,vbar ) ) \
        ##+ (1-theta) * facet_integral( dot(pb0I*n,vbar) )
        ##- (1-theta) * facet_integral( 2*nu*inner(u0, sym(grad(vbar))*n) ) \
        ##- (1-theta) * facet_integral( alpha/he * 2*nu*dot(ubar0,vbar)) \
        ##+ (1-theta) * facet_integral( dot(pb0I*n,vbar) ) \
        ##+ (1-theta) * facet_integral(dot(ubar0,n)*qbar)  \
        ##+ (1-theta) * facet_integral( beta*he/(nu+1) * pbar0 * qbar )

    #Q, S = Form(Q), Form(S)
    #ssc = StokesStaticCondensation(mesh, A, G, GT, B, Q, S)