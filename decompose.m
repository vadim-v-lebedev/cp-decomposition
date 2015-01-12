function decompose(s1, s2, s3, s4, r)
    w = load('weights.txt');
    w = reshape(w, [s1 s2 s3 s4]);
    %save('w.mat', 'w');
    options.Initialization = @cpd_rnd;
    [U,asd] = cpd(w, r, options);
    if 0
        for i = 1:4
            size(U{i})
        end
    end
    delta_w = cpdgen(U) - w;
    err = norm(delta_w(:)) / norm(w(:))
    names = ['x', 'y', 'c', 'n'];
    for i = 1:4
        temp = U{i};
        save(['f_', names(i), '.txt'], 'temp', '-ascii');
    end
end
