using Flux, Statistics

"""
    sim!(model, tws, deltats, xtrain, ytrain, batch_size, l_rate, l2, number_data)

Compute the loss, accuracy at times tws and MSD between two two different times (tws, tws .+ deltats)

- `model` refers to a `Chain` (Flux type)


"""
function sim!(model, tws, deltats, xtrain, ytrain, batch_size, learning_rate, l2, number_data)
    
    parameters = Flux.params(model);
    totalparams = sum(length, parameters)

    ##Auxiliary functions    
    sqnorm(x) = sum(abs2, x)
    penalty(lambda) = lambda*sum(sqnorm, parameters)
    loss(x, y) = Flux.crossentropy(model(x), y)  + penalty(l2)  
    accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
    ######

    
    total_times = [tw .+ deltats for tw in tws]
    msd = [[] for i in 1:length(total_time)];
    init_params = [[] for i in 1:length(total_time)];
    loss_ev = []
    acc = []
   
    
    
    (m, n) = size(xtrain);
    glob_time = 0
    opt = Descent(learning_rate)

    while glob_time < total_times[end][end]
        for minibatch in Iterators.partition(1:number_data,batch_size)
            glob_time += 1
            data = [(xtrain[:, batch], ytrain[:, batch])]
            Flux.train!(loss, parameters, data, opt)
            
            if glob_time in tws
                index = findfirst(x-> x == glob_time, tws)
                init_params[index] = deepcopy(vcat([vcat([parameters...][i]...) 
                                                    for i in 1:length([parameters...])]...))
                push!(loss_ev, loss(xtrain, ytrain))
                push!(acc, accuracy(xtrain, ytrain))
                println(glob_time)
            end
          
            
            for tw_index in 1:length(total_times)
                if glob_time in total_times[tw_index]
                    parameters_t =  deepcopy(vcat([vcat([parameters...][i]...) for i in 1:length([parameters...])]...))
                    push!(msd[time_index], sum(abs2, parameters_t .- init_params[tw_index])/totalparams)
                end
            end
        end
    end
        
    loss_ev, acc, msd
end
    
    
