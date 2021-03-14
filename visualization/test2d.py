import plot_2d

x=['Linear', 'Ridge', 'Lasso']
group=['PCA','ANOVA']
y=[[0.85, 0.92, 1],[0.83, 0.99, 8]]

xy={
    'PCA':[0.85, 0.92, 1],
    'ANOVA' :[0.83, 0.99, 8]
}
plot_2d.bar_2d(x,xy)