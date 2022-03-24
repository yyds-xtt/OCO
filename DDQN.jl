using Flux

# ----------------------------------------------
# build the structure of the DDQN

struct DNN
    # each layer of the network contain the number of nodes
    layer_features::Array{Int}
    network::Flux.Chain
    target_network::Flux.Chain
    function DNN(layer_features)
        network = []
        for i = 1:(length(layer_features)-1)
            push!(network,Flux.Dense(layer_features[i],layer_features[i+1]))
        end
        new(layer_features,Flux.Chain(network...),Flux.Chain(copy(network)...))
    end
end

function (ddqn::DNN)(x)
    return ddqn.network(x)
end

# -----------------------------------------------

# define network's param
Num_BS = 20
NUm_TimeSlot = 30
Frequency_Edge = [5,10]
Size_Task = [60,90]
CPU_Task = [3,4,5]
k = 1*10^-27
W = 10
I = 2*10^-13
sigma = 0.7
CPU_Freq_Device = [1,3]
Trans_Freq_Device = [200,400]
F = 2
P = 2
E0 = 3000
Pool = 1024

# Build the entire network

layer2 = Int(floor((Num_BS+4)^(2/3)*(Num_BS*P+F)^(1/3)))
layer3 = Int(floor((Num_BS+4)^(1/3)*(Num_BS*P+F)^(2/3)))

# define the online network
OnlineLatencyNetwork = DNN([Num_BS+4,layer2,layer3,Num_BS*P+F])
OnlineEnnergyNetwork = DNN([Num_BS+4,layer2,layer3,Num_BS*P+F])
# define the target network
TargetLatencyNetwork = DNN([Num_BS+4,layer2,layer3,Num_BS*P+F])
TargetEnnergyNetwork = DNN([Num_BS+4,layer2,layer3,Num_BS*P+F])

# define exprience pool

