
using WaveletsEvaluation
using WaveletsEvaluation.Filterbanks
using WaveletsEvaluation.InfiniteVectors
using Test, Random, LinearAlgebra

P = 80

rng = MersenneTwister(3000)
jumpfunction(x) = (   (-0.5 < x < 0.5) ? 1.0 : 0.0) + ((-0.25 < x < .75) ? 1.0 : 0.0)
characteristicfunction(x) = (0<x<1) ? 1.0 : 0.0
randomfunction(x) = rand(rng)

@testset "$(rpad("Invertibility of dwtstep",P))"  begin
    for N in 10:10:1000
        t = LinRange(-1,1,N)
        for f in (sin, characteristicfunction, randomfunction)
            x = map(f,t)
            for w in (DWT.IMPLEMENTED_WAVELETS...,)
                fb = Filterbank(w)
                for bound in (DWT.perbound,)
                    y = dwtstep(x, fb, bound)
                    xx = idwtstep(y..., fb, bound)
                    @test (norm(xx-x)) < 1e-10
                end
            end
        end
    end
end

@testset "$(rpad("Invertibility of dwt",P))"  begin
    T = Float64
    for n in 1:10
        N = 2^n
        t = LinRange(-1,1,N)
        for f in (sin, characteristicfunction, randomfunction)
            x = map(f,t)
            for w in (DWT.IMPLEMENTED_WAVELETS...,)
                for bound in (DWT.perbound, )
                    for L in 1:n
                        y = DWT.dwt(x, w, bound, L)
                        xx = DWT.idwt(y, w, bound, L)
                        @test (norm(xx-x)) < 1e-10

                        y = DWT.dwt(x, Dual, w, bound, L)
                        xx = DWT.idwt(y, Dual, w, bound, L)
                        @test (norm(xx-x)) < 1e-8
                    end
                end
            end
        end
    end
end

@testset "$(rpad("Invertibility of full_dwt using constant function (periodic)",P))"  begin
    for l in 0:10
        x0 = ones(1<<l)
        for w in DWT.IMPLEMENTED_WAVELETS
            x1 = full_dwt(x0, w, DWT.perbound)
            y  = full_idwt(x1, w, DWT.perbound)
            @test (norm(x0-y)< 1e-10)
        end
    end
end

@testset "$(rpad("Invertibility of full_dwt (periodic)",P))"  begin
    for l in 0:7
        for p in 0:9
            x0 = [Float64(i^p) for i in 1:1<<l]; x0/=sum(x0)
            for i in p:9
                w = DWT.DaubechiesWavelet{i+1,Float64}()
                offset = (max([support_length(side, kind, w) for side in (Primal, Dual) for kind in (scaling, DWT.wavelet)]...,))
                x1 = full_dwt(x0, w, DWT.perbound)
                y  = full_idwt(x1, w, DWT.perbound)
                d = abs.(y-x0)
                @test (sum(d[offset+1:end-offset])<1e-10)
            end
        end
    end
end

for w in (db2, cdf24)
    @testset "Unscaled wavelet evaluation" begin
        @test evaluate_in_dyadic_points(Primal, scaling, w, 0, 0, 0;scaled=true)*sqrt(2)≈
            evaluate_in_dyadic_points(Primal, scaling, w, 1, 0, 1;scaled=true)
        @test evaluate_in_dyadic_points(Primal, scaling, w, 0, 0, 0;scaled=false)≈
            evaluate_in_dyadic_points(Primal, scaling, w, 1, 0, 1;scaled=false)
        @test evaluate_in_dyadic_points(Primal, scaling, w, 0, 0, 10;scaled=true)*2sqrt(2)≈
            evaluate_in_dyadic_points(Primal, scaling, w, 3, 0, 13;scaled=true)
        @test evaluate_in_dyadic_points(Primal, scaling, w, 0, 0, 10;scaled=false)≈
            evaluate_in_dyadic_points(Primal, scaling, w, 3, 0, 13;scaled=false)

        @test evaluate_in_dyadic_points(Primal, wavelet, w, 0, 0, 0;scaled=true)*sqrt(2)≈
            evaluate_in_dyadic_points(Primal, wavelet, w, 1, 0, 1;scaled=true)
        @test evaluate_in_dyadic_points(Primal, wavelet, w, 0, 0, 0;scaled=false)≈
            evaluate_in_dyadic_points(Primal, wavelet, w, 1, 0, 1;scaled=false)
        @test evaluate_in_dyadic_points(Primal, wavelet, w, 0, 0, 10;scaled=true)*2sqrt(2)≈
            evaluate_in_dyadic_points(Primal, wavelet, w, 3, 0, 13;scaled=true)
        @test evaluate_in_dyadic_points(Primal, wavelet, w, 0, 0, 10;scaled=false)≈
            evaluate_in_dyadic_points(Primal, wavelet, w, 3, 0, 13;scaled=false)

        @test WaveletsEvaluation.DWT.evaluate_periodic_scaling_basis_in_dyadic_points(Primal, w, [1,0,0,0,0,0,0,0],4,scaled=true)[1:4]≈
            (WaveletsEvaluation.DWT.evaluate_periodic_scaling_basis_in_dyadic_points(Primal, w, [1,0,0,0],3,scaled=true)*sqrt(2))[1:4]
        @test WaveletsEvaluation.DWT.evaluate_periodic_scaling_basis_in_dyadic_points(Primal, w, [1,0,0,0,0,0,0,0],4,scaled=false)[1:4]≈
            WaveletsEvaluation.DWT.evaluate_periodic_scaling_basis_in_dyadic_points(Primal, w, [1,0,0,0],3,scaled=false)[1:4]
        @test WaveletsEvaluation.DWT.evaluate_periodic_scaling_basis_in_dyadic_points(Primal, w, [1,0,0,0,0,0,0,0],4,scaled=true)≈
            (WaveletsEvaluation.DWT.evaluate_periodic_scaling_basis_in_dyadic_points(Primal, w, [1,0,0,0,0,0,0,0],4,scaled=false)*2sqrt(2))

        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0,0,0,0,0],4,scaled=true)[1:8]≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0],3,scaled=true)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,1,0,0,0,0,0,0],4,scaled=true)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,1,0,0],4,scaled=true)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,0,1,0,0,0,0,0],4,scaled=true)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,0,1,0],4,scaled=true)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,0,0,1,0,0,0,0],4,scaled=true)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,0,0,1],4,scaled=true)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [
            0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],4,scaled=true)[1:8]*sqrt(2)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [
            0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],5,scaled=true)[1:8]

        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0,0,0,0,0],4,scaled=false)[1:8]≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0],3,scaled=false)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,1,0,0,0,0,0,0],4,scaled=false)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,1,0,0],4,scaled=false)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,0,1,0,0,0,0,0],4,scaled=false)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,0,1,0],4,scaled=false)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,0,0,1,0,0,0,0],4,scaled=false)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [0,0,0,1],4,scaled=false)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [
            0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],4,scaled=false)[1:8]≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [
            0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],5,scaled=false)[1:8]

        # Constant should be scale invariant
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0],2,scaled=true)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0],2,scaled=false)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0,0,0,0,0],3,scaled=false)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0,0,0,0,0],3,scaled=false)
        @test WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],4,scaled=false)≈
            WaveletsEvaluation.DWT.evaluate_periodic_wavelet_basis_in_dyadic_points(Primal, w, [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],4,scaled=false)
    end
end
