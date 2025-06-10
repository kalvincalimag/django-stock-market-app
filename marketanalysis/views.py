from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import EmailMessage, send_mail
from predictly import settings
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth import authenticate, login, logout
from django.views.generic import View
from django.contrib.auth.decorators import login_required

from . tokens import generate_token
from rest_framework.views import APIView
from .forms import StockForm
from .models import Stock
from .forms import PasswordResetForm, FeedbackForm

from .constants import (
    FAQ_GLOSSARY, 
    MARKET_GLOSSARY
)

from .plotting import (
    generate_closing_prices_plot,
    generate_crossover_plot, 
    generate_prediction_vs_actual_plot,
    generate_weekly_forecast_plot,
    determine_cross_signal
)
from .services import (
    fetch_stock_data,
    get_company_info,
    calculate_sma_dataframe,
    prepare_lstm_data,
    get_or_train_lstm_model,
    make_predictions,
    generate_future_predictions
)

import pandas as pd
import datetime
import os
import requests
import json

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def home(request):
    return render(request, "authentication/signup.html")

def password_reset_done(request):
    return render(request, "authentication/password_reset_done.html")

def password_reset_complete(request):
    return render(request, "authentication/password_reset_complete.html")

def faq(request):
    glossary = {
        "What is Predictly Stock Trend Predictor?": "Predictly Stock Trend Predictor is a web application that uses machine learning to analyze historical stock data and provide predictions on future stock price trends. It aims to assist users in making informed investment decisions.",
        "How does Predictly Stock Trend Predictor work?": "The application uses historical stock data and machine learning algorithms to identify patterns and trends. It then makes predictions based on this analysis to provide insights into potential future stock price movements.",
        "Is Predictly Stock Trend Predictor suitable for beginners?": "Yes, Predictly is designed to cater to both beginners and experienced investors. It provides user-friendly tools and visualizations to help users understand and analyze stock data.",
        "Is registration required to use the application?": "Yes, you need to register for an account to use Predictly Stock Trend Predictor. Registration allows you to access personalized features, including the watchlist and prediction services.",
        "How can I reset my password if I forget it?": "If you forget your password, you can use the 'Forgot Password?' option on the Sign-in Page. This will guide you through the password recovery process, including sending a password reset email to your registered email address.",
        "Can I track multiple stocks with Predictly?": "Yes, you can create a personalized watchlist to track multiple stocks simultaneously. The Watchlist Page allows you to monitor the performance of your selected stocks at a glance.",
        "What is the 'Golden Cross' and 'Death Cross' on the Automated Crossover Page?": "The 'Golden Cross' occurs when the Simple Moving Average (SMA) 100 surpasses the SMA 200, indicating a potential upswing in the stock's performance. The 'Death Cross' is identified when the stock's current price falls below both the SMA 100 and SMA 200, suggesting a potential downward trend.",
        "How often is the stock data updated on the Last Trade Prices Page?": "Stock data on the Last Trade Prices Page is typically updated regularly, providing you with the most recent information. The frequency of updates may vary based on the data source and market conditions.",
        "Can I use Predictly on my mobile device?": "Yes, Predictly is designed to be responsive and accessible on both desktop and mobile devices. You can use it on your smartphone or tablet for convenient access on the go.",
        "Where does Predictly Stock Trend Predictor source its stock data?": "Predictly Stock Trend Predictor sources its stock data from reputable financial data providers, primarily IEX Cloud and Yahoo Finance. These sources provide reliable and up-to-date information on stock prices, market trends, and related data.",
        "Why does Predictly use data from IEX Cloud and Yahoo Finance?": "We utilize data from IEX Cloud and Yahoo Finance because they are well-established data providers known for their accuracy and comprehensive coverage of financial markets. This ensures that the data used for predictions and analysis is of high quality.",
        "Is the stock data on Predictly real-time or delayed?": "The availability of real-time or delayed data may vary depending on the specific data source and market conditions. Predictly strives to provide the most up-to-date data available from IEX Cloud and Yahoo Finance, but it's essential to be aware that some data may have a slight delay.",
        "Are there any limitations on the availability of certain stock data on Predictly?": "Data availability may vary based on the specific stock and the data sources. In some cases, certain stocks or data points may not be available due to limitations imposed by IEX Cloud or Yahoo Finance. Predictly strives to provide the most comprehensive data possible.",
        "Can I rely on the data accuracy provided by Predictly for my investment decisions?": "While Predictly aims to provide accurate and reliable data, it's important to remember that all investments carry risks, and data should be used as one of several factors in your investment decisions. Always conduct thorough research and consult with financial professionals before making investment choices.",
        "Can I share my watchlist with others?": "Currently, watchlists are designed for personal use and cannot be shared with other users. Your watchlist is private and customizable based on your preferences.",
        "Are there any additional resources or educational materials for stock market beginners?": "Yes, we provide educational resources and articles in the Market Glossary section of the application. These materials can help beginners understand stock market concepts and strategies.",
        "Can I connect my brokerage account to Predictly for real-time data and trading?": "Predictly is primarily a stock analysis and prediction tool. It does not support real-time trading or direct connections to brokerage accounts. Users should use their brokerage platforms for trading.",
        "What do I do if I encounter technical issues while using the application?": "If you encounter technical issues or have questions about using Predictly, you can reach out to our customer support team for assistance. We're here to help you with any challenges you may face.",
    }

    return render(request, "pages/faq.html", {"glossary": glossary})

def market_glossary(request):
    glossary = {
    "Stock": "A share in the ownership of a company. When you own a company's stock, you own a piece of the company.",
    "Stock Exchange": "A regulated marketplace where stocks and other securities are bought and sold.",
    "Bull Market": "A market characterized by rising prices. It's a time when investors are optimistic about the future performance of the market.",
    "Bear Market": "A market condition marked by declining stock prices, typically caused by pessimism and economic downturns.",
    "Volatility": "The degree of variation in a stock's price over time, often associated with higher risk and uncertainty.",
    "Market Capitalization": "The total value of a company's outstanding shares of stock, calculated by multiplying the stock price by the number of shares.",
    "Liquidity": "The ease with which an asset or security can be bought or sold in the market without affecting its price.",
    "Liquidity Risk": "The risk that an asset cannot be quickly bought or sold in the market without significantly affecting its price.",
    "Blue Chip Stocks": "Stocks of well-established, financially stable, and reputable companies known for their reliability and stability.",
    "Technical Analysis": "The study of historical price and volume data to make predictions about future price movements.",
    "Fundamental Analysis": "The evaluation of a company's financial health, including earnings, assets, and liabilities, to determine its stock's intrinsic value.",
    "Short Selling": "A strategy where an investor borrows and sells a stock they don't own, anticipating a price decline to buy it back at a lower price.",
    "Market Sentiment": "The overall mood and attitude of investors toward the market or a particular asset, which can influence buying and selling decisions.",
    "Day Trading": "A trading strategy where positions are opened and closed within the same trading day, capitalizing on short-term price fluctuations.",
    "Price-to-Earnings Ratio": "A valuation metric that compares a company's stock price to its earnings per share, helping assess its relative value.",
    "Limit Order": "An order to buy or sell a security at a specific price or better, ensuring the trade is executed at a certain price or not at all.",
    "Futures Contract": "A standardized financial contract that obligates the buyer to purchase and the seller to sell a specified asset at a predetermined future date and price.",
    "Resistive Support Levels": "Price levels on a stock chart where a stock tends to encounter difficult rising or difficulty falling.",
    "Capital Gain": "The profit realized when an asset is sold for a higher price than its original purchase price.",
    "Leverage Ratio": "A measure of a company's debt relative to its equity, indicating its level of financial leverage and risk.",
    "Value Investing": "An investment strategy that involves selecting stocks trading at prices lower than their intrinsic value, based on fundamentals, with the expectation of long-term growth.",
    "Candlestick Chart": "A type of price chart used in technical analysis that displays price movements in a visually informative manner, resembling candlesticks.",
    "Dead Cat Bounce": "A temporary, small recovery in the price of a declining asset, often followed by a further decline.",
    "Hedging": "A risk management strategy where an investor uses financial instruments like options or futures to offset potential losses in another investment.",
    "Recession": "A significant and sustained decline in economic activity characterized by reduced consumer spending, investment, and employment.",
    "Risk-Adjusted Return": "A measure of investment performance that takes into account the level of risk taken to achieve a certain return, often using metrics like the Sharpe ratio.",
    "Market Timing": "An investment strategy that involves trying to predict the future movements of financial markets to buy and sell assets at optimal times.",
    "Stock Broker": "A licensed professional or firm that facilitates the buying and selling of securities on behalf of investors.",
    "Bullish": "Positive sentiment or outlook on a stock or the overall market.",
    "Bearish": "Negative sentiment or outlook on a stock or the overall market.",
    "Rally": "A period of sustained increases in stock prices.",
    "Portfolio Diversification": "Spreading investments across different asset classes or securities to reduce risk.",
    "Dividend": "A portion of a company's earnings paid to its shareholders on a per-share basis.",
    "Penny Stock": "A low-priced, highly speculative stock typically trading for less than $5 per share.",
    "Market Correction": "A decline of at least 10% from a recent high in a stock or index.",
    "Liquidity Provider": "An entity or individual that offers to buy or sell assets in the financial market, enhancing liquidity by facilitating transactions.",
    "Market Order": "A type of order to buy or sell a security at the current market price, ensuring immediate execution but not a specific price.",
    "Option Contract": "A financial derivative that grants the holder the right, but not the obligation, to buy or sell an underlying asset at a predetermined price within a specified timeframe.",
    "Hedge Fund": "A pooled investment fund that employs various strategies to generate returns for its investors, often with a focus on high-risk and high-reward opportunities.",
    "Earnings Per Share": "A financial metric that represents a company's profit allocated to each outstanding share of common stock, providing insight into its profitability.",
    "Margin Trading": "A strategy where investors borrow funds to buy securities, using the purchased assets as collateral, increasing potential returns but also risks.",
    "Inflation": "The gradual increase in the general price level of goods and services, reducing the purchasing power of a currency over time.",
    "Yield": "The return on an investment, typically expressed as a percentage, taking into account dividends, interest, or other income generated.",
    "Index Fund": "A type of mutual fund or exchange-traded fund (ETF) designed to replicate the performance of a specific market index, providing diversified exposure to the underlying assets."
}
    
    if request.user.is_authenticated:
                fname = request.user.first_name
    else:
        fname = ""
    
    return render(request, "pages/market_glossary.html", {'fname': fname, 'glossary': glossary})

def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('home')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('home')

        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('home')
        
        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('home')

        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric!!")
            return redirect('home')

        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.is_active = False  
        myuser.save()
        messages.success(request, "Your Account has been created succesfully! Please check your email to confirm your email address in order to activate your account.")
        
        # Welcome Email Start
        subject = "🌟 Welcome to Predictly!"
        message = f"Hey {myuser.first_name}, welcome to Predictly! 🚀\n\nGet ready to ride the wave of stock trends with us. Your financial journey starts here!\n\nHowever, first things first, we've sent you a confirmation email, please check your inbox to confirm your email address.\n\nWe can't wait to see you delve into the stock world.\n\n- The Predictly Team 📈"

        from_email = settings.EMAIL_HOST_USER
        to_list = [myuser.email]

        html_message = f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        margin: 20px;
                        padding: 20px;
                        background-color: #f4f4f4;
                        color: #333;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        background-color: #fff;
                        padding: 40px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}
                    h1 {{
                        color: #FF4B4B;
                    }}
                    p {{
                        margin-bottom: 20px;
                    }}
                    strong {{
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Welcome to Predictly!</h1>
                    <p>Hey <strong>{myuser.first_name}</strong>, welcome to Predictly! 🚀</p>
                    <p>Get ready to ride the wave of stock trends with us. Your financial journey starts here!</p>
                    <p>However, first things first, we've sent you a confirmation email, please check your inbox to confirm your email address.</p>
                    <p>We can't wait to see you delve into the stock world.</p>
                    <p>- The Predictly Team 📈</p>
                </div>
            </body>
            </html>
        """

        send_mail(
            subject,
            '',  
            from_email,
            to_list,
            html_message=html_message,  
            fail_silently=True,
        )
        
        # Welcome Email End

        
        # User Email Confirmation Start
        
        current_site = get_current_site(request)
        email_subject = "🚀 Confirm your Email @ Predictly"

        message2 = render_to_string('email_templates/user_email_confirmation.html', {
            'name': myuser.first_name,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
            'token': generate_token.make_token(myuser),
        })

        email = EmailMessage(
            email_subject,
            '',
            settings.EMAIL_HOST_USER,
            [myuser.email],
        )
        email.content_subtype = 'html'  
        email.body = message2  
        email.fail_silently = True
        email.send()
        
        # User Email Confirmation End
        
        return redirect('signin') 
        
    return render(request, "authentication/signup.html")

def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        
        user = authenticate(username=username, password=pass1)
        
        if user is not None:
            login(request, user)
            fname = user.first_name
            return redirect('last-trade-prices', ticker_symbol='AAPL')  

        else:
            messages.error(request, "Bad Credentials!!")
            return redirect('signin')

    return render(request, "authentication/signin.html")

def signout(request):
    logout(request)
    messages.success(request, "Logged Out Successfully!!")
    return redirect('home')

def activate(request,uidb64,token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        myuser = User.objects.get(pk=uid)
    except (TypeError,ValueError,OverflowError,User.DoesNotExist):
        myuser = None

    if myuser is not None and generate_token.check_token(myuser,token):
        myuser.is_active = True
        # user.profile.signup_confirmation = True
        myuser.save()
        login(request,myuser)
        messages.success(request, "Your Account has been activated!!")
        return redirect('signin')
    else:
        return render(request,'activation_failed.html')
    
def forgot_password(request):
    if request.method == 'POST':
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data['email']
            associated_users = User.objects.filter(email=data)
            if associated_users.exists():
                for user in associated_users:
                    token = default_token_generator.make_token(user)
                    uid = urlsafe_base64_encode(force_bytes(user.pk))
                    subject = '🔒 Password Reset Requested'
                    email_template_name = 'email_templates/reset_password_email.html'
                    c = {
                        'email': user.email,
                        'domain': request.META['HTTP_HOST'],
                        'site_name': 'your site',
                        'uid': uid,
                        'user': user,
                        'token': token,
                        'protocol': 'http',
                    }
                    email = render_to_string(email_template_name, c)
                    send_mail(subject, '', 'predictlyapp@gmail.com', [user.email], fail_silently=False, html_message=email)
                return redirect("password_reset_done")
            else:
                return redirect("password_reset_done")
    else:
        form = PasswordResetForm()
    return render(request=request, template_name="authentication/forgot_password.html", context={"form": form})

def reset_password(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
        
    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            new_password = request.POST.get('new_password')
            confirm_password = request.POST.get('confirm_password')
            if new_password == confirm_password:
                user.set_password(new_password)
                user.save()
                return redirect("password_reset_complete") 
            else:
                return render(request, 'authentication/password_reset.html', {'error_message': "Passwords do not match."})
        else:
            return render(request, 'authentication/password_reset.html', {'uidb64': uidb64, 'token': token})
    else:
        return render(request, 'authentication/password_reset.html', {'error_message': "The link is no longer valid."})

def feedback_view(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            feedback = form.cleaned_data['feedback']

            sender_email = 'predictlyfeedback@gmail.com'

            email_content = render_to_string('email_templates/user_feedback_email.html', {'email': email, 'feedback': feedback})

            send_mail(
                '✉️ New Customer Feedback',
                '',
                sender_email,
                ['predictlyfeedback@gmail.com'],
                html_message=email_content,
                fail_silently=False,
            )

            return JsonResponse({'message': 'Feedback submitted successfully'})
        else:
            return JsonResponse({'message': 'Invalid form data'}, status=400)
    else:
        form = FeedbackForm()

    return render(request, 'feedback_template.html', {'form': form})

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

@login_required  
def my_watchlist(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            stock = form.save(commit=False)
            stock.user = request.user 
            stock.save()
            messages.success(request, "Stock has been added!")
            return redirect('my_watchlist')
    else:
        form = StockForm()  
        ticker = Stock.objects.filter(user=request.user)  
        output = []

        for ticker_item in ticker:
            api_request = requests.get(
                f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_item}&apikey={ALPHA_VANTAGE_API_KEY}")
            try:
                api = json.loads(api_request.content)
                if 'Global Quote' in api:
                    stock_data = api['Global Quote']
                    stock_data['pk'] = ticker_item.pk
                    stock_data['added_at'] = ticker_item.added_at
                    output.append(stock_data)
                else:
                    output.append({'symbol': str(ticker_item), 'error': 'Data not found: API limit reached.'})
            except Exception as e:
                output.append({'symbol': str(ticker_item), 'error': str(e)})
             
        print(output)

        if request.user.is_authenticated:
            fname = request.user.first_name.capitalize()
        else:
            fname = ""
        
        return render(request, 'pages/my_watchlist.html', {'form': form, 'ticker': ticker, 'output': output, 'fname': fname,})

@login_required
def delete(request, stock_id):
    try:
        stock = Stock.objects.get(pk=stock_id)
        if stock.user == request.user: 
            stock.delete()
            messages.success(request, "Stock has been deleted!")
        else:
            messages.error(request, "You can only delete your own stocks!")
    except Stock.DoesNotExist:
        messages.error(request, "Stock not found!")
    
    return redirect('my_watchlist')



closing_prices_plot_lock = threading.Lock()
crossover_plot_lock = threading.Lock()
prediction_plot_lock = threading.Lock()
weekly_forecast_plot_lock = threading.Lock()

def determine_cross_signal(sma_dataframe):
    golden_cross = sma_dataframe['SMA100'].iloc[-1] > sma_dataframe['SMA200'].iloc[-1]
    death_cross = sma_dataframe['Close'].iloc[-1] < sma_dataframe['SMA100'].iloc[-1] and sma_dataframe['Close'].iloc[-1] < sma_dataframe['SMA200'].iloc[-1]

    if golden_cross:
        return 'A Golden Cross has been detected'
    elif death_cross:
        return 'A Death Cross has been detected'
    else: 
        return 'No significant cross has been detected'

def generate_closing_prices_plot(stockdataframe):
    with closing_prices_plot_lock:
        fig = px.line(stockdataframe, x=stockdataframe.index, y='Close')
        plot_div = fig.to_html(full_html=False)
        return plot_div

def generate_crossover_plot(sma_dataframe):
    with crossover_plot_lock:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sma_dataframe['Date'], y=sma_dataframe['Close'], mode='lines', name='Closing Prices', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=sma_dataframe['Date'], y=sma_dataframe['SMA100'], mode='lines', name='SMA 100', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=sma_dataframe['Date'], y=sma_dataframe['SMA200'], mode='lines', name='SMA 200', line=dict(color='green')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', legend_title='Indicators')
        # Determine Golden Cross or Death Cross
        cross_signal = determine_cross_signal(sma_dataframe)
        fig.add_annotation(text=f"{cross_signal}", xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False, font=dict(color="black", size=12), bgcolor="red", opacity=0.5)
        plot_div = fig.to_html(full_html=False)
        return plot_div

def generate_prediction_vs_actual_plot(dates, y_test_flat, y_predicted_flat, future_prediction=False):
    with prediction_plot_lock:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_test_flat, mode='lines', name='Actual Prices' , line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=y_predicted_flat, mode='lines', name='Predicted Prices', line=dict(color='red')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', legend_title='Prices')
        title = "Weekly Forecast" if future_prediction else "Prediction vs. Actual"
        return fig

def generate_weekly_forecast_plot(dates, y_predicted_flat):
    with weekly_forecast_plot_lock:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=y_predicted_flat, mode='lines', name='Weekly Forecast', line=dict(color='green')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', legend_title='Prices')

        # Calculate overall trend direction
        trend_direction = 'Bearish' if y_predicted_flat[0] > y_predicted_flat[-1] else 'Bullish'
        
        # Add annotation
        fig.add_annotation(text=f'{trend_direction} Trend', xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False, font=dict(size=12, color='black'), bgcolor="red", opacity=0.5)
        return fig

class LastTradePricesAPIView(View):
    def get(self, request, ticker_symbol=None):
        
        ticker_symbol = request.GET.get('ticker_symbol', 'AAPL')

        if ticker_symbol is None:
            ticker_symbol = 'AAPL'

        if ticker_symbol:
            start_date = '2010-01-01'
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')
            
            try:
                stockdataframe = yf.download(ticker_symbol, start=start_date, end=end_date)
            except:
                context = {
                    'ticker_symbol': ticker_symbol, 
                    'error_message': "Invalid Ticker",
                }
                return render(request, 'pages/last_trade_prices_template.html', context)
            
            if stockdataframe.empty:
                context = {
                    'error_message': "Invalid Ticker",
                }
                return render(request, 'pages/last_trade_prices_template.html', context)

            stock_info_mapping = {
                "NMS": "NASDAQ",
                "NYQ": "NYSE",
            }
            
            closing_prices_plot = generate_closing_prices_plot(stockdataframe)
            raw_data_summary = stockdataframe.describe()
            ticker_info = yf.Ticker(ticker_symbol)
            # company_name = ticker_info.info['longName']
            stock_exchange = stock_info_mapping.get(ticker_info.fast_info['exchange'], ticker_info.fast_info['exchange'])
            
            if request.user.is_authenticated:
                fname = request.user.first_name
            else:
                fname = ""

            context = {
               # 'company_name': company_name,
                'stock_exchange': stock_exchange,
                'raw_data_summary': raw_data_summary,
                'closing_prices_plot': closing_prices_plot,
                'stockdataframe': stockdataframe.to_html(classes='table table-bordered table-striped'),
                'ticker_symbol': ticker_symbol, 
                'fname': fname,  
            }
            
        return render(request, 'pages/last_trade_prices_template.html', context)   

class AutomatedCrossoverAPIView(APIView):
    def get(self, request, ticker_symbol=None):
        
        ticker_symbol = request.GET.get('ticker_symbol', 'AAPL')
        
        if ticker_symbol is None:
            ticker_symbol = 'AAPL'

        if ticker_symbol:
            start_date = '2010-01-01'
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')

            try:
                stockdataframe = yf.download(ticker_symbol, start=start_date, end=end_date)
            except:
                context = {
                    'ticker_symbol': ticker_symbol, 
                    'error_message': "Invalid Ticker",
                }
                return render(request, 'pages/crossover_data_template.html', context)
            
            if stockdataframe.empty:
                context = {
                    'error_message': "Invalid Ticker",
                    'ticker_symbol': ticker_symbol,
                }
                return render(request, 'pages/crossover_data_template.html', context)

            stock_info_mapping = {
                "NMS": "NASDAQ",
                "NYQ": "NYSE",
            }

            ticker_info = yf.Ticker(ticker_symbol)
            # company_name = ticker_info.info['longName']
            stock_exchange = stock_info_mapping.get(ticker_info.fast_info['exchange'], ticker_info.fast_info['exchange'])

            simple_moving_avg_100 = stockdataframe['Close'].rolling(100).mean()
            simple_moving_avg_200 = stockdataframe['Close'].rolling(200).mean()

            sma_dataframe = pd.DataFrame({
                'Date': stockdataframe.index,
                'Close': stockdataframe['Close'],
                'SMA100': simple_moving_avg_100,
                'SMA200': simple_moving_avg_200
            })

            # Generate crossover plot
            crossover_plot = generate_crossover_plot(sma_dataframe)
            # Determine Cross Signal
            cross_signal = determine_cross_signal(sma_dataframe)

            if request.user.is_authenticated:
                fname = request.user.first_name
            else:
                fname = ""

            context = {
                # 'company_name': company_name,
                'stock_exchange': stock_exchange,
                'crossover_plot': crossover_plot,
                'sma_dataframe': sma_dataframe.to_html(classes='table table-bordered table-striped'),
                'ticker_symbol': ticker_symbol, 
                'fname': fname, 
                'cross_signal': cross_signal, 
            }

            return render(request, 'pages/crossover_data_template.html', context)     

class PredictionVsActualAPIView(APIView):
    def get(self, request, ticker_symbol=None):
                
        ticker_symbol = request.GET.get('ticker_symbol', 'AAPL')

        if ticker_symbol is None:
            ticker_symbol = 'AAPL'

        if ticker_symbol:
            start_date = '2010-01-01'
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')

            try:
                stockdataframe = yf.download(ticker_symbol, start=start_date, end=end_date)
            except:
                context = {
                    'ticker_symbol': ticker_symbol, 
                    'error_message': "Invalid Ticker",
                }
                return render(request, 'pages/prediction_vs_actual_template.html', context)

            if stockdataframe.empty:
                context = {
                    'error_message': "No data available for the specified ticker",
                }
                return render(request, 'pages/prediction_vs_actual_template.html', context)

            stock_info_mapping = {
                "NMS": "NASDAQ",
                "NYQ": "NYSE",
            }

            ticker_info = yf.Ticker(ticker_symbol)
            stock_exchange = stock_info_mapping.get(ticker_info.fast_info['exchange'], ticker_info.fast_info['exchange'])

            # Data Pre-processing Phase 1 - Splitting into Training & Testing
            training_70 = pd.DataFrame(stockdataframe['Close'][0:int(len(stockdataframe) * 0.70)])
            testing_30 = pd.DataFrame(stockdataframe['Close'][int(len(stockdataframe) * 0.70): int(len(stockdataframe))])

            # Data Pre-processing Phase 2 - Scaling/Normalization
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(training_70)

            # Model Integration
            model = load_model('marketanalysis/keras_models/keras_model.keras')

            # Testing Part
            past_100_days = training_70.tail(100)
            final_dataframe = pd.concat([past_100_days, testing_30], ignore_index=True)
            input_data = scaler.fit_transform(final_dataframe)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100: i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)
            scaler = scaler.scale_
            
            # Reverse Scaling
            scale_factor = 1 / scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor

            # Flatten the arrays
            y_test_flat = y_test.flatten()
            y_predicted_flat = y_predicted.flatten()
        
            dates = stockdataframe.index[int(len(stockdataframe) * 0.70):]
            prediction_vs_actual_plot = generate_prediction_vs_actual_plot(dates, y_test_flat, y_predicted_flat)
                        
            # Generate Weekly Forecast Plot
            future_dates = pd.date_range(start=dates[-1], periods=8, freq='D')[1:]  # Next 7 days
            future_x_test = input_data[-100:]  # Using last 100 days for prediction
            future_predictions = []

            for _ in range(7):
                future_prediction = model.predict(np.array([future_x_test]))[0][0]
                future_predictions.append(future_prediction)
                future_x_test = np.roll(future_x_test, -1)  # Roll input data to add new prediction
                future_x_test[-1][0] = future_prediction

            # Reverse Scaling for future predictions
            future_predictions = np.array(future_predictions) * scale_factor

            weekly_forecast_plot = generate_weekly_forecast_plot(future_dates, future_predictions)
            
            # For the Front End Interpretation
            future_trend = 'Bearish Trend' if future_predictions[0] > future_predictions[-1] else 'Bullish Trend'
            
            context = {
                'stock_exchange': stock_exchange,
                'prediction_vs_actual_plot': prediction_vs_actual_plot.to_html(full_html=False),
                'weekly_forecast_plot': weekly_forecast_plot.to_html(full_html=False),
                'ticker_symbol': ticker_symbol,
                'future_trend': future_trend, 
            }

            return render(request, 'pages/prediction_vs_actual_template.html', context)
