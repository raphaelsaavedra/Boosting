using RCall, DataFrames

function initialize(label_a::Int, label_b::Int)::Tuple{DataFrames.DataFrame,DataFrames.DataFrame}
    @rput label_a
    @rput label_b
    R"""
    load('mnist_train.rda')
    mnist = subset(mnist_train, mnist_train[,1] %in% c(label_a, label_b))
    idx_non_zero = which(colSums(mnist) != 0)
    mnist = mnist[,idx_non_zero]    
    mnist_means = aggregate(mnist[,-1], by = list(mnist[,1]), FUN = mean)
    """
    @rget mnist
    @rget mnist_means
    return mnist, mnist_means
end

function baseClassifier_j(mnist_means::DataFrames.DataFrame, pixel::Int, j::Int64, correct::Int64) # Classificador que ve de qual media o valor do pixel esta mais proximo
    if abs(pixel - mnist_means[1,j]) <= abs(pixel - mnist_means[2,j])           
        label = mnist_means[1,1] 
        return label, correct == label
    else
        label = mnist_means[2,1] 
        return label, correct == label
    end
end

function findMostCommon(vector::Vector{Int64}, a::Int, b::Int)
    count_a = 0; count_b = 0
    for el in vector
        if el == a
            count_a += 1
        else
            count_b += 1
        end
    end
    if count_a > count_b
        return a
    else
        return b
    end
end

function main(label_a::Int, label_b::Int, ϵ::Float64, γ::Float64)
    #label_a = 4; label_b = 5; ϵ = 0.15; γ = 0.1    
    mnist, mnist_means = initialize(label_a, label_b);
    
    d = size(mnist, 2) - 1
    n = size(mnist, 1)
    w = ones(n)
    η = γ
    T = convert(Int, ceil(2/γ^2 * log(1/ϵ)))
    println("T = ", T)
    corrects = falses(n)
    labels = Vector{Int}(n)
    labels_per_classifier = Matrix{Int}(n, T)
    for t in 1:T
        j = 1
        soma = 0.0
        while soma < 0.5 + γ
            p = w / sum(w)
            soma = 0.0
            for i in 1:n
                label, correct_label = baseClassifier_j(mnist_means, mnist[i, j + 1], j + 1, mnist[i, 1])
                labels[i] = label
                corrects[i] = correct_label            
            end

            soma = sum(p .* corrects)

            j += 1
            if j > d
                println("Nao foi encontrado weak learner valido para a iteracao ", t)
                return 
                break

            end
        end    
        labels_per_classifier[:,t] = labels
        # Encontrado weak learner correspondente ao j-esimo baseClassifier
        # Agora, atualizando os pesos
        for i in 1:n
            w[i] = w[i] * exp(η * !corrects[i]) # Aumenta o peso de quem foi classificado errado
        end    
        p = w / sum(w)
        if t % 10 == 0
            println("t = ", t, " de ", T) 
        end            
    end

    correct_final_label = mnist[:X5]
    final_label = Vector{Int64}(n);
    acertos = falses(n)
    for i in 1:n
        final_label[i] = findMostCommon(labels_per_classifier[i,:], label_a, label_b)
        acertos[i] = final_label[i] == correct_final_label[i]
    end

    percentual_acertos = sum(acertos)/length(acertos)
    println(percentual_acertos)
end

@time main(4, 5, 0.05, 0.1)

# Retornar tambem as outras coisas que pede no trabalho
# Ao inves de rodar o loop ate T necessariamente, posso rodar um while so enquanto o percentual de acertos nao for alcançado