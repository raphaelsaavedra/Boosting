using RCall, DataFrames

"""Load characters data"""
function initialize(label_a::Int, label_b::Int)::Tuple{DataFrames.DataFrame,DataFrames.DataFrame}
    @rput label_a
    @rput label_b
    R"""
    load('mnist_train.rda')
    mnist = subset(mnist_train, mnist_train[,1] %in% c(label_a, label_b))
    mnist_means = aggregate(mnist[,-1], by = list(mnist[,1]), FUN = mean)
    """
    @rget mnist
    @rget mnist_means
    return mnist, mnist_means
end

"""Single pixel classifier based on mean pixel value"""
function baseClassifier_j(mnist_means::DataFrames.DataFrame, pixel::Int, j::Int, correct::Int)
    if abs(pixel - mnist_means[1,j]) <= abs(pixel - mnist_means[2,j])
        label = mnist_means[1,1]
    else
        label = mnist_means[2,1]
    end
    return label, correct == label
end

"""Return which label is the most common in a vector"""
function findMostCommon(vector::Vector{Int}, a::Int, b::Int)
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
    mnist, mnist_means = initialize(label_a, label_b)

    d = size(mnist, 2) - 1
    n = size(mnist, 1)
    w = ones(n)
    η = γ
    T = convert(Int, ceil(2/γ^2 * log(1/ϵ)))
    info("T = ", T)
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
                label, correct_label = baseClassifier_j(mnist_means, Int64(mnist[i, j + 1]), j + 1, Int64(mnist[i, 1]))
                labels[i] = label
                corrects[i] = correct_label
            end

            soma = sum(p .* corrects)

            j += 1
            if j > d
                warn("No valid weak learner found for iteration ", t)
            end
        end
        labels_per_classifier[:, t] = labels
        # Found weak learner corresponding to j-th baseClassifier
        # Update weights
        for i in 1:n
            w[i] = w[i] * exp(η * !corrects[i]) # Increase weights for those that were wrongly classified
        end
        p = w / sum(w)
        if t % 10 == 0
            info("t = ", t, " of ", T)
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
    info("Correct ratio: ", percentual_acertos)
end

@time main(0, 1, 0.2, 0.1)

# Retornar tambem as outras coisas que pede no trabalho
# Ao inves de rodar o loop ate T necessariamente, posso rodar um while so enquanto o percentual de acertos nao for alcançado
