using RCall, DataFrames

label_a = 4
label_b = 7
@rput a
@rput b
R"""
load("mnist_train.rda")
mnist = subset(mnist_train, mnist_train[,1] %in% c(label_a, label_b))
mnist_means = aggregate(mnist[,-1], by = list(mnist[,1]), FUN = mean)
"""
@rget mnist
@rget mnist_means

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


n = size(mnist, 2) - 1
d = size(mnist, 1)
w = ones(d)
γ = 0.05
η = 0.15
T = convert(Int, 1e2)
corrects = falses(d)
labels = Vector{Int}(d)
labels_per_classifier = Matrix{Int}(d, T)
for t in 1:T
    j = 1
    soma = 0
    while soma < 0.5 + γ
        p = w / sum(w)
        soma = 0.0
        for i in 1:d
            label, correct_label = baseClassifier_j(mnist_means, mnist[i, j + 1], j + 1, mnist[i, 1])
            labels[i] = label
            corrects[i] = correct_label            
        end

        soma = sum(p .* corrects)

        j += 1
        if j > n
            println("Nao foi encontrado weak learner valido para a iteracao ", t)
            break
        end
    end    
    labels_per_classifier[:,t] = labels
    # Encontrado weak learner correspondente ao j-esimo baseClassifier
    # Agora, atualizando os pesos
    for i in 1:d
        w[i] = w[i] * exp(η * !corrects[i]) # Aumenta o peso de quem foi classificado errado
    end    
    p = w / sum(w)
end

correct_final_label = mnist[:X5]
final_label = Vector{Int64}(d);
acertos = falses(d)
for i in 1:d
    final_label[i] = findMostCommon(labels_per_classifier[i,:], label_a, label_b)
    acertos[i] = final_label[i] == correct_final_label[i]
end

sum(acertos)/length(acertos)



