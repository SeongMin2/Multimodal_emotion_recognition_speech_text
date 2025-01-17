import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout_ratio, device):
        super().__init__()

        # 어차피 여러겹으로 쌓을것이 아니기 때문에 head의 수 만큼 차원을 나눌 필요가 없음

        # assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim  # 임베딩 차원
        self.n_heads = n_heads  # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
        self.head_dim = hidden_dim  # head의 dim을 나눠줄 필요가 없거든
        # self.head_dim = hidden_dim  // n_heads  # 각 헤드(head)에서의 임베딩 차원

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)  # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)  # Key 값에 적용될 FC 레이어
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)  # Value 값에 적용될 FC 레이어

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q: [batch_size, query_len, hidden_dim]
        # K: [batch_size, key_len, hidden_dim]
        # V: [batch_size, value_len, hidden_dim]

        # hidden_dim → n_heads X head_dim 형태로 변형
        # n_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q: [batch_size, n_heads, query_len, head_dim]
        # K: [batch_size, n_heads, key_len, head_dim]
        # V: [batch_size, n_heads, value_len, head_dim]

        # Attention Energy 계산
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy: [batch_size, n_heads, query_len, key_len]

        # 마스크(mask)를 사용하는 경우
        if mask is not None:
            # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
            energy = energy.masked_fill(mask == 0, -1e10)

        # 어텐션(attention) 스코어 계산: 각 단어에 대한 확률 값
        attention = torch.softmax(energy, dim=-1)

        # attention: [batch_size, n_heads, query_len, key_len]

        # 여기에서 Scaled Dot-Product Attention을 계산
        x = torch.matmul(self.dropout(attention), V)

        # x: [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x: [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hidden_dim)

        # x: [batch_size, query_len, hidden_dim]

        x = self.fc_o(x)

        # x: [batch_size, query_len, hidden_dim]

        return x, attention

class Cross_attention2(nn.Module):
    def __init__(self, config):
        super(Cross_attention2, self).__init__()
        self.hidden_dim = config.attention_emb  # 임베딩 차원
        self.n_heads = config.n_heads  # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
        self.head_dim = config.attention_emb  # head의 dim을 나눠줄 필요가 없거든
        self.dropout_ratio = config.dropout_ratio

        '''
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)  # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)  # Key 값에 적용될 FC 레이어
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        '''

        self.fc_q = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.n_heads)])
        self.fc_k = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.n_heads)])
        self.fc_v = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.n_heads)])

        self.dropout = nn.Dropout(self.dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))  # .to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]

        attention_head = []
        for i in range(self.n_heads):
            Query = self.fc_q[i](query)
            Key = self.fc_q[i](key)
            Value = self.fc_q[i](value)

            # Query: [batch_size, query_len, hidden_dim]
            # Key: [batch_size, key_len, hidden_dim]
            # Value: [batch_size, value_len, hidden_dim]

            alpha = torch.matmul(Query, Key.transpose(1,2)) / self.scale
            # energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

            if mask is not None:
                alpha = alpha.masked_fill(mask == 0, -1e9)

            alpha = alpha / torch.sqrt(torch.Tensor([self.hidden_dim]))

            attn = torch.softmax(alpha, dim=-1)

            attn = self.dropout(attn)

            if i == 0:
                attention_head = torch.matmul(attn, Value)
            else:
                attention_head = torch.cat((attention_head, torch.matmul(attn, Value)), dim=-1)

        return attention_head


class Cross_attention(nn.Module):
    def __init__(self, config):
        super(Cross_attention, self).__init__()


        self.hidden_dim = config.attention_emb  # 임베딩 차원
        self.n_heads = config.n_heads  # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
        self.head_dim = int(config.attention_emb / config.n_heads) # head의 dim을 나눠줄 필요가 없거든
        self.dropout_ratio = config.dropout_ratio


        self.fc_q = nn.Linear(self.hidden_dim, self.hidden_dim)  # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(self.hidden_dim, self.hidden_dim)  # Key 값에 적용될 FC 레이어
        self.fc_v = nn.Linear(self.hidden_dim, self.hidden_dim)  # Value 값에 적용될 FC 레이어

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.dropout = nn.Dropout(self.dropout_ratio)


        self.scale = torch.nn.parameter.Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])), requires_grad=False)  # .to(config.device)

    def Scaled_Dot_Product_attention(self, query, key, value, mask):

        alpha = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale
        '''
        if mask is not None:
            # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
            #alpha = alpha.masked_fill(mask == 0, -1e10)
            pass
        '''
        if mask is not None:
            # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
            alpha = alpha.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e10)

        # 어텐션(attention) 스코어 계산: 각 단어에 대한 확률 값
        alpha = torch.softmax(alpha, dim=-1)

        # alpha: [batch_size, n_heads, query_len, key_len]

        # 여기에서 Scaled Dot-Product Attention을 계산
        x = torch.matmul(self.dropout(alpha), value)

        # x: [batch_size, n_heads, query_len, head_dim]

        return x


    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.shape[0]

        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q: [batch_size, query_len, hidden_dim]
        # K: [batch_size, key_len, hidden_dim]
        # V: [batch_size, value_len, hidden_dim]

        # hidden_dim → n_heads X head_dim 형태로 변형
        # n_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q: [batch_size, n_heads, query_len, head_dim]
        # K: [batch_size, n_heads, key_len, head_dim]
        # V: [batch_size, n_heads, value_len, head_dim]

        x = self.Scaled_Dot_Product_attention(Q, K, V, attn_mask)

        # x: [batch_size, n_heads, query_len, head_dim]

        ##############################################
        #            Concatenation of Heads          #
        ##############################################
        x = x.permute(0, 2, 1, 3).contiguous()

        # x: [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hidden_dim)

        # x: [batch_size, query_len, hidden_dim]

        x = self.fc(x)

        return x
