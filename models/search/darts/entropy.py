def compute_arch_entropy(self, dim=-1):
        alpha = self.arch_parameters()[0]
        prob = F.softmax(alpha, dim=dim)
        log_prob = F.log_softmax(alpha, dim=dim)
        entropy = - (log_prob * prob).sum(-1, keepdim=False)
        return entropy