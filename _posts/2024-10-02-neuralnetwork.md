---
title: Neural network bằng tay 🤚
date: 2024-10-01 21:37:00 +0700
categories: [Chương 1]
tags: [kiến thức]     # TAG names should always be lowercase
---

# Chúng ta sẽ học gì?

Tại blogs này, tôi với bạn sẽ cùng tìm hiểu về Neural network, một kiến thức nền tảng để xây dựng bất kì mô hình Deep learning nào. Chúng ta sẽ cùng nhau giải thích được những khái niệm liên quan tới neural network, tự tính tay các nút ở trong mạng này và ứng dụng việc vào việc xây dựng một mô hình Neural network đơn giản nhận diện được chữ viết tay MNIST.

Blogs được chia thành từng phần như nhau:
1. Các khái niệm liên quan tới neural network
2. Xây dựng một mô hình Neural Network đơn giản
3. Ứng dụng vào giải quyết bài toán nhận diện chữ viết tay MNIST

Nào, chúng ta cùng bắt đầu thôi!

# 1. Khái niệm liên quan tới Neural Network.

Neural Network - Mạng thần kinh, nghe đã có một chút gì đó liên quan tới bộ não con người rồi đúng không? Thật sự là vậy, các mô hình Neural network được lấy cảm hứng từ các tế bào thần kinh (Neural) trong bộ não con người. Bạn có thể nhìn thấy hình ảnh dưới đây để hiểu rõ hơn.

![alt text](/images/neuralnetwork/braincell.png)

Bạn có thể thấy ở hình ảnh trên, các dòng diện trong não chạy vào các chân Dendrites, đi qua tính toán ...

Ngoài lề một chút, tuy rằng cho dù bạn có thể thấy các mô hình ngôn ngữ lớn hiện nay như chatGPT, Claude hay Perplexity cực kì thông minh đi chăng nữa. Chúng vẫn chỉ là một khối tính toán khổng lồ, **mô phỏng chỉ một phần nhỏ** trí lực của bộ não chúng ta. Tại sao tôi lại nói như vậy?  Vì theo người bố già trong lĩnh vực AI, Andrew Ng, cũng đã khẳng định điều tương tự: *Đến chính con người chúng ta còn không biết não bộ hoạt động như thế nào.* Việc có thể phát triển lên AGI (Artificial General Intelligence), có thể tạo ra một loại máy móc có thể tự suy nghĩ, lập luận những thứ mới mà con người không biết tới. Tuy nhiên, tương lai đó vẫn còn rất xa vời, và hi vọng rằng chúng ta có thể chiêm nghiệm được nó trong một tương lai không xa.

Trở lại vấn đề, lí do tôi nói điều trên vì tôi muốn nhấn mạnh rằng: Neural network **mô phỏng một phần rất nhỏ** trong bộ não con người. Vì nó chỉ bao gồm toàn các phép tính toán ma trận, giải tích đạo hàm, và xác suất thống kê - 3 trụ cột chính xây dựng lên các mô hình Deep learning. Tuy nghe sợ như vậy, nhưng cũng sẽ dễ hiểu hơn nếu các bạn có thể liên hệ được những kiến thức đó tới thực tế trong cuộc sống. Mình sẽ nói kĩ hơn ở các phần sau.

Vậy thì các khái niệm liên quan tới neural network bao gồm những thứ gì? Mình sẽ nói tóm gọn lại những khái niệm chính như sau:

1. Neural Network: Một loại mô hình mô phỏng chức năng bộ não của con người: Cho một đầu vào thích hợp thì sẽ cho một đầu ra tương ứng sau khi tính toán qua các lớp layer.
1. Layer: Một tập hợp các nút (Nodes), có nhiệm vụ đóng gói các nodes trong các lớp neural network lại với nhau.
2. Nodes: Các nút ở trong một layer, bạn có thể hình dung rằng các nút này sẽ nhận các đầu vào từ nút khác, tính toán rồi lại trả lại một giá trị đầu ra.

Cơ bản là vậy trước đã, trong bài toán neural network thì cho dù về sau cũng phức tạp đến mấy, suy cho cùng cũng là một bài toán ánh xạ từ một đầu vào cho trước tới một đầu ra. Nếu hiểu cực kì đơn giản, các mô hình LLM tiên tiến nhất hiện nay như chatGPT, Claude,... cũng không khác gì một bài toán dự đoán giá nhà cơ bản trong machine learning vậy cả. Chỉ là tính toán giữa các lớp layer có thể trừu tượng, hoặc nhiều công thức tính toán hơn thôi.

Và một số thuật ngữ căn bản trong Neural network mà bạn sẽ cần phải sử dụng tới, chưa cần phải hiểu vội đâu ^^ :

1. Feedforward: "Feed" là cho ăn, còn "Forward" là tiến lên. Bạn có thể hiểu cụm từ này là việc bạn cho ăn model ăn đầu vào của mình, các input giữa các lớp sẽ dần dần *tiến lên* đến các nút output và trả về đầu ra cho bạn.
2. Loss function: Giống như việc dạy học một đứa trẻ vậy, để xem đứa trẻ ấy có học hỏi được tốt hay không, bạn cần phương pháp đánh giá và điều chỉnh học tập của đứa trẻ ấy. Loss function là một hàm mất mát, nó chỉ ra rằng độ tệ, chưa tốt của một mô hình khi đang thực hiện một chức năng nào đó. Bạn cần phải làm thế nào để sao cho hàm loss function có giá trị nhỏ nhất có thể.
2. Backpropagation: "Back" là ngược lại, còn "Propagation" là truyền theo. Vậy bạn cần làm gì để loss function nhỏ nhất có thể? Các bạn hẳn đã quen với đạo hàm rồi, khi muốn tìm vận tốc bé nhất trên một quãng đường, thì bạn phải đạo hàm phương trình quãng đường. Loss funciton cũng vậy, bạn phải đạo hàm để tìm ra điểm cực tiểu *tốt nhất* mà bạn kiếm được. 

    Bật mí trước cho bạn rằng, ví dụ như mô hình ngôn ngữ Llama 3.1 mới nhất mà Facebook phát triển hiện nay, con số nodes (tạm hiểu là vậy) lên tới 405 tỉ nodes !!! Bạn ngồi tính tay 3 cái đạo hàm thôi là đã thấy chán rồi đúng không, đây còn phải tính tận 405 tỉ cái nữa thì ...

    Nhưng, Backpropagation sinh ra để giải quyết về vấn đề đạo hàm. Vì các nodes được lên kết với nhau qua từng layer, việc bạn lấy đạo hàm của một node trong layer trước, và tận dụng đạo hàm đó truyền ngược lại để tính các đạo hàm của các nodes phía sau nó. Đó chính là backpropagation.

Tạm thời đó là đủ những kiến thức mà mình muốn bạn biết đến, chi tiết về từng kiến thức một mình sẽ bổ sung dần về sau. Mình sẽ chia nhỏ từng blogs một cho ngắn thôi để các bạn đỡ bị ngợp @@. Và thế nhé, xin gặp các bạn ở các post sau.