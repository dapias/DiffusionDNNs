using JLD
using MLDatasets

#include("../src/training.jl")

# load full training set
train_x, train_y = CIFAR10.traindata(Float32)
# load full test set
test_x,  test_y  = CIFAR10.testdata(Float32);

ytrain =  Flux.onehotbatch(train_y, 0:9);

tws = round.(Int64,(exp10.(range(0, stop = log10(500000), length = 28))));
deltats = Int64.(ceil.(exp10.(range(0, stop = 6, length = 35))));

#tws = round.(Int64,(exp10.(range(0, stop = log10(500), length = 10))));
#deltats = Int64.(ceil.(exp10.(range(0, stop = 3, length = 15))));


###Baity-Jesi's SmallNet
model = Chain(
  Conv((5,5), 3=>10, relu),
    MaxPool((2,2)),
  Conv((5,5), 10=>20, relu),
    MaxPool((2,2)),
 x -> reshape(x, :, size(x, 4)),
  Dense(500, 100, relu),
    Dense(100, 10), softmax)

model(train_x[:,:,:,1:1])
println("evalué el modelo")

##Auxiliary functions    
sqnorm(x) = sum(abs2, x)
loss(x, y) = Flux.crossentropy(model(x), y)   
accuracy(x, y) = mean(Flux.onecold(model(x)) .== Flux.onecold(y))
######

batch_size = 100
learning_rate = 0.01; 
(m, n, z, number_data) = size(train_x)

total_times = [tw .+ deltats for tw in tws]
msd = [[] for i in 1:length(total_times)];
init_params = [[] for i in 1:length(total_times)];
loss_ev = []
acc = []

global glob_time = 0
opt = Descent(learning_rate)

parameters = Flux.params(model);
totalparams = sum(length, parameters)

while glob_time <= total_times[end][end]
    for minibatch in Iterators.partition(1:number_data,batch_size)
        global glob_time += 1
        data = [(train_x[:,:,:,minibatch], ytrain[:, minibatch])]
        Flux.train!(loss, parameters, data, opt)
        
        if glob_time in tws
            index = findfirst(x-> x == glob_time, tws)
            init_params[index] = deepcopy(vcat([vcat([parameters...][i]...) 
                                                for i in 1:length([parameters...])]...))
        end
        
        if glob_time in deltats
            push!(loss_ev, loss(train_x, ytrain))
            push!(acc, accuracy(train_x, ytrain))
            #println(glob_time)
        end
        
        for tw_index in 1:length(total_times)
            if glob_time in total_times[tw_index]
                parameters_t =  deepcopy(vcat([vcat([parameters...][i]...) for i in 1:length([parameters...])]...))
                push!(msd[tw_index], sum(abs2, parameters_t .- init_params[tw_index])/totalparams)

                println(glob_time)
                save("../data/cifar10.jld", "loss", loss_ev, "accuracy", acc, "msd", msd, "deltats", deltats, "tws", tws)
            end 
        end
    end    
end

